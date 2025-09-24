import json
import math
import os
from typing import Dict, Literal, Tuple
from typing import Optional

import numpy as np
import sys


# ---------- 基础工具 ----------

def unpack_bits_uint8(x: np.ndarray, hid_dim: int, bitorder: Literal["little","big"]="little") -> np.ndarray:
    """
    将 uint8 打包的比特按位解压为 0/1 的 int8，返回形状 [..., hid_dim]。
    x 的最后一维是 ceil(hid_dim/8)。
    """
    y = np.unpackbits(x, axis=-1, bitorder=bitorder)   # [..., pack*8]
    return y[..., :hid_dim].astype(np.int8)            # 截到 hid_dim


def to_pm1_int8(bits01: np.ndarray) -> np.ndarray:
    """把 0/1 转成 -1/+1（int8）"""
    return np.where(bits01 == 0, np.int8(-1), np.int8(1))


def group_reduce_q(q_pm1: np.ndarray, group: int, Hk_expect: Optional[int],
                   reduce: Literal["sum","mean"]="sum") -> np.ndarray:
    batchSize, one, headQ, hidDim = q_pm1.shape
    if headQ % group != 0:
        raise ValueError("headQ 不是 group 的整数倍")
    headK = headQ // group
    q_tmp = q_pm1.reshape(batchSize, 1, headK, group, hidDim).astype(np.int32)
    if reduce == "sum":
        return q_tmp.sum(axis=3)
    elif reduce == "mean":
        return q_tmp.mean(axis=3)
    else:
        raise ValueError("reduce 必须是 sum 或 mean")



def compute_score(q_grouped: np.ndarray, k_pm1: np.ndarray) -> np.ndarray:
    """
    维度转换 + 乘法求和（沿 hidDim 维）：得到 score（相似度，越大越好）。
    输入:
      q_grouped: [batchSize,1,headK,hidDim] (int32)
      k_pm1    : [batchSize,seqLen,headK,hidDim] (int8)
    输出:
      score    : [batchSize,headK,1,seqLen] (int32)
    """
    batchSize, one, headK, hidDim = q_grouped.shape
    if one != 1:
        raise ValueError("q_grouped 的第 2 维必须为 1")
    q_t = np.transpose(q_grouped, (0, 2, 1, 3))  # [batchSize,headK,1,hidDim]
    k_t = np.transpose(k_pm1,     (0, 2, 1, 3))  # [batchSize,headK,seqLen,hidDim]
    score = np.einsum('bhqd,bhsd->bhqs', q_t, k_t, optimize=True).astype(np.int32)
    return score


def chunk_reduce_lastdim(x: np.ndarray, chunk_size: int = 16,
                         reduce: Literal["sum","mean","max"]="sum") -> Tuple[np.ndarray, int]:
    """
    在最后一维按定长 chunk 聚合。使用零填充到整数块。
    返回 (y, ChunkS)，y 的形状是 [..., ChunkS]。
    """
    *prefix, seqLen = x.shape
    ChunkS = math.ceil(seqLen / chunk_size)
    target = ChunkS * chunk_size
    if target != seqLen:
        pad = [(0, 0)] * (x.ndim - 1) + [(0, target - seqLen)]
        x = np.pad(x, pad, mode="constant", constant_values=0)
    x = x.reshape((*prefix, ChunkS, chunk_size))
    if reduce == "sum":
        y = x.sum(axis=-1)
    elif reduce == "max":
        y = x.max(axis=-1)
    elif reduce == "mean":
        counts = np.full((ChunkS,), chunk_size, dtype=np.int32)
        if seqLen % chunk_size != 0:
            counts[-1] = seqLen % chunk_size
        y = x.sum(axis=-1) / counts.reshape((1,) * len(prefix) + (ChunkS,))
    else:
        raise ValueError(f"unsupported reduce: {reduce}")
    return y.astype(np.int32), ChunkS


def topk_lastdim(x: np.ndarray, k: int, largest: bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    沿最后一维做 top-k，返回 (values, indices)，按 values 降序/升序排序。
    """
    if not (1 <= k <= x.shape[-1]):
        raise ValueError(f"topK 必须在 1..{x.shape[-1]} 范围内，给定 {k}")
    axis = x.ndim - 1
    kth = k - 1
    if largest:
        part_idx = np.argpartition(-x, kth=kth, axis=axis)[..., :k]
        part_val = np.take_along_axis(x, part_idx, axis=axis)
        order = np.argsort(-part_val, axis=axis)
    else:
        part_idx = np.argpartition(x, kth=kth, axis=axis)[..., :k]
        part_val = np.take_along_axis(x, part_idx, axis=axis)
        order = np.argsort(part_val, axis=axis)
    topk_idx = np.take_along_axis(part_idx, order, axis=axis)
    topk_val = np.take_along_axis(part_val, order, axis=axis)
    return topk_val, topk_idx


def save_bin(path: str, arr: np.ndarray) -> None:
    """裸写为 .bin（无 header）。"""
    arr.tofile(path)


def save_meta_json(path: str, meta: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ---------- 主流程（按 1~6 步） ----------

def hamming_topk_pipeline_bitpacked(
    qhash_packed_u8: np.ndarray,     # [batchSize,1,headQ,ceil(hidDim/8)]  按位打包
    khash_packed_u8: np.ndarray,     # [batchSize,seqLen,headK,ceil(hidDim/8)]  按位打包
    hid_dim: int,                    # 解压后的 D（bit 数）
    group: int,                      # headQ = group * headK
    topK: int,                       # 沿 ChunkS 取 top-k
    chunk_size: int = 16,
    chunk_reduce: Literal["sum","mean","max"] = "sum",
    take_largest: bool = True,       # True=score最大；若要“距离最小”，看下方注释
    bitorder: Literal["little","big"] = "little",
    input_save_dir: Optional[str] = None,      # 若非 None，保存 qhash/khash/topk_idx
    output_save_dir: Optional[str] = None      # 若非 None，保存 qhash/khash/topk_idx
) -> Dict[str, np.ndarray]:
    """
    返回：
      {
        "score":        [batchSize,headK,1,seqLen]         int32
        "score_chunk":  [batchSize,headK,1,ChunkS]    int32
        "topk_vals":    [batchSize,headK,1,topK]      int32
        "topk_idx":     [batchSize,headK,1,topK]      int32   # ChunkS 维索引
      }
    """
    # 形状检查（打包态）
    if qhash_packed_u8.ndim != 4 or khash_packed_u8.ndim != 4:
        raise ValueError("qhash/khash 必须是 4 维（打包状态）")
    B1, one, headQ, packDq = qhash_packed_u8.shape
    B2, seqLen, headK, packDk = khash_packed_u8.shape
    if one != 1:
        raise ValueError(f"qhash 的第 2 维必须为 1，得到 {one}")
    if B1 != B2:
        raise ValueError(f"batchSize 不一致: qhash batchSize={B1}, khash batchSize={B2}")
    if packDq != packDk:
        raise ValueError(f"打包后的最后一维不一致: {packDq} vs {packDk}")
    if headQ % group != 0:
        raise ValueError(f"headQ={headQ} 不是 group={group} 的整数倍")
    if topK <= 0:
        raise ValueError("topK 必须 > 0")
    if seqLen % chunk_size != 0:
        raise ValueError(f"seqLen={seqLen} 不是 chunk_size={chunk_size} 的整数倍")

    # ---- Step 1: 按位解压 → 0/1 → ±1 ----
    q_bits01 = unpack_bits_uint8(qhash_packed_u8, hid_dim, bitorder=bitorder)  # [batchSize,1,headQ,hidDim] int8
    k_bits01 = unpack_bits_uint8(khash_packed_u8, hid_dim, bitorder=bitorder)  # [batchSize,seqLen,headK,hidDim] int8
    q_pm1 = to_pm1_int8(q_bits01)   # [-1,+1]
    k_pm1 = to_pm1_int8(k_bits01)   # [-1,+1]

    # ---- Step 2: q 按 group 累加 → [batchSize,1,headK,hidDim] ----
    q_grouped = group_reduce_q(q_pm1, group=group, Hk_expect=headK, reduce="mean")  # int32

    # ---- Step 3 & 4: 维度变换 & 乘法求和 → score:[batchSize,headK,1,seqLen] ----
    score = compute_score(q_grouped, k_pm1)  # 越大越相似

    # 【可切到“汉明距离最小”】：
    #   dist = ((group * hid_dim) - score) // 2      # [batchSize,headK,1,seqLen] 越小越近
    #   后续把 score 替换为 dist，并设置 take_largest=False 即可。

    # ---- Step 5: 沿 seqLen 分块（长度 16），→ [batchSize,headK,1,ChunkS] ----
    score_chunk, ChunkS = chunk_reduce_lastdim(score, chunk_size=chunk_size, reduce=chunk_reduce)
    if topK > ChunkS:
        raise ValueError(f"topK={topK} 不能大于 ChunkS={ChunkS}")

    # ---- Step 6: 在 ChunkS 维做 top-k ----
    topk_vals, topk_idx = topk_lastdim(score_chunk, k=topK, largest=take_largest)

    # ---- 落盘（可选）----
    if input_save_dir is not None:
        os.makedirs(input_save_dir, exist_ok=True)
        # 原始（打包态）直接写出
        save_bin(os.path.join(input_save_dir, "input_qhash.bin"), qhash_packed_u8)
        save_bin(os.path.join(input_save_dir, "input_khash.bin"), khash_packed_u8)
        # topK 索引：int32
        save_bin(os.path.join(output_save_dir, "golden_topk_index.bin"), topk_idx.astype(np.int32))
        # 元信息
        meta = {
            "qhash_packed": {"shape": list(qhash_packed_u8.shape), "dtype": str(qhash_packed_u8.dtype)},
            "khash_packed": {"shape": list(khash_packed_u8.shape), "dtype": str(khash_packed_u8.dtype)},
            "hid_dim": hid_dim,
            "group": group,
            "seqLen": seqLen,
            "headK": headK,
            "headQ": headQ,
            "chunk_size": chunk_size,
            "chunk_reduce": chunk_reduce,
            "take_largest": take_largest,
            "topK": topK,
            "score_shape": list(score.shape),
            "score_chunk_shape": list(score_chunk.shape),
            "topk_index_shape": list(topk_idx.shape),
            "bitorder": bitorder
        }
        save_meta_json(os.path.join(input_save_dir, "meta.json"), meta)

    return {
        "score": score,                         # [batchSize,headK,1,seqLen]
        "score_chunk": score_chunk,             # [batchSize,headK,1,ChunkS]
        "topk_vals": topk_vals,                 # [batchSize,headK,1,topK]
        "topk_idx": topk_idx.astype(np.int32),  # [batchSize,headK,1,topK]（ChunkS 维索引）
    }


# ---------- 最小可运行示例 ----------
if __name__ == "__main__":
    # 配置
    batchSize = int(sys.argv[1])
    seqLen = int(sys.argv[2])
    headQ = int(sys.argv[3])
    headK = int(sys.argv[4])
    hidDim = int(sys.argv[5])
    topK = int(sys.argv[6])
    
    Group = headQ // headK
    assert(headQ == Group * headK)
    assert(hidDim % 8 == 0)  # 为简化示例，要求 hidDim 是 8 的倍数

    chunk_size = 16
    bitorder = "big"   # 若你的打包是 big-endian bits，可改为 "big"

    # 随机生成“打包态”的输入：最后一维是 ceil(hidDim/8)
    packD = (hidDim + 7) // 8
    rng = np.random.default_rng(0)
    qhash_packed = rng.integers(0, 256, size=(batchSize, 1, headQ, packD), dtype=np.uint8)
    khash_packed = rng.integers(0, 256, size=(batchSize, seqLen, headK, packD), dtype=np.uint8)

    out = hamming_topk_pipeline_bitpacked(
        qhash_packed_u8=qhash_packed,
        khash_packed_u8=khash_packed,
        hid_dim=hidDim,
        group=Group,
        topK=topK,
        chunk_size=chunk_size,
        chunk_reduce="max",     # 可选 "sum"/"mean"/"max"
        take_largest=True,      # 若需“距离最小”，见上方注释
        bitorder=bitorder,
        input_save_dir="./input",  # 若不落盘，改为 None
        output_save_dir="./output"  # 若不落盘，改为 None
    )

    print("score:", out["score"].shape)               # [batchSize,headK,1,seqLen]
    print("score_chunk:", out["score_chunk"].shape)   # [batchSize,headK,1,ChunkS]
    print("topk_vals:", out["topk_vals"].shape)       # [batchSize,headK,1,topK]
    print("topk_idx:", out["topk_idx"].shape)         # [batchSize,headK,1,topK]

    # print("score_chunk: ", out["score_chunk"])
    # print("topk_idx: ", out["topk_idx"])
