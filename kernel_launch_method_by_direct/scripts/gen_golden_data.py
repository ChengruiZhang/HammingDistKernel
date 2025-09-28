import json
import math
import os
from typing import Dict, Literal, Tuple
from typing import Optional

import numpy as np
import sys

import numpy as np
import math
from typing import Literal, Tuple, Dict, Optional

def _lut16_popcount() -> np.ndarray:
    """返回 uint16 -> popcount 的查表（长度 65536，dtype=uint8）。"""
    # 用 8-bit LUT 叠两次也行；这里直接构 16-bit LUT，简单直接
    lut = np.empty(1 << 16, dtype=np.uint8)
    for i in range(len(lut)):
        lut[i] = bin(i).count("1")
    return lut

_LUT16 = _lut16_popcount()

def _chunk_reduce_lastdim(x: np.ndarray, chunk_size: int = 16,
                          reduce: Literal["sum","mean","max"]="sum") -> Tuple[np.ndarray, int]:
    """在最后一维按定长 chunk 聚合；必要时末尾零填充。返回 (y, ChunkS)。"""
    *prefix, L = x.shape
    ChunkS = math.ceil(L / chunk_size)
    target = ChunkS * chunk_size
    if target != L:
        pad = [(0, 0)] * (x.ndim - 1) + [(0, target - L)]
        x = np.pad(x, pad, mode="constant", constant_values=0)
    x = x.reshape((*prefix, ChunkS, chunk_size))
    if reduce == "sum":
        y = x.sum(axis=-1)
    elif reduce == "max":
        y = x.max(axis=-1)
    elif reduce == "mean":
        counts = np.full((ChunkS,), chunk_size, dtype=np.int32)
        if L % chunk_size != 0:
            counts[-1] = L % chunk_size
        y = x.sum(axis=-1) / counts.reshape((1,) * len(prefix) + (ChunkS,))
    else:
        raise ValueError("reduce 必须是 'sum'/'mean'/'max'")
    return y.astype(np.int32), ChunkS

def _topk_lastdim(x: np.ndarray, k: int, largest: bool=False) -> Tuple[np.ndarray, np.ndarray]:
    """沿最后一维做 top-k，返回 (values, indices)。默认取最小（largest=False）。"""
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

def hamming_topk_from_uint16_packed(
    qhash_packed: np.ndarray,      # (B, 1, headQ, packD), uint16
    khash_packed: np.ndarray,      # (B, seqLen, headK, packD), uint16
    headQ: int,
    headK: int,
    topK: int,
    chunk_size: int = 16,
    chunk_reduce: Literal["sum","mean","max"] = "sum",
) -> Dict[str, np.ndarray]:
    """
    输出：
      - dist:        (B, headK, 1, seqLen)      int32  —— 已沿 Group 累加的总 Hamming
      - dist_chunk:  (B, headK, 1, ChunkS)      int32  —— chunk 压缩后
      - topk_vals:   (B, headK, 1, topK)        int32  —— chunk 层面的最小 topK 值
      - topk_idx:    (B, headK, 1, topK)        int32  —— chunk 索引（0..ChunkS-1）
    说明：默认 Hamming 越小越相似，因此 top-k 取最小（largest=False）
    """
    if qhash_packed.dtype != np.uint16 or khash_packed.dtype != np.uint16:
        raise ValueError("qhash_packed/khash_packed 必须是 uint16")
    B1, one, hQ_in, packDq = qhash_packed.shape
    B2, seqLen, hK_in, packDk = khash_packed.shape
    if one != 1: raise ValueError("qhash_packed 的第2维必须为 1")
    if B1 != B2: raise ValueError("batchSize 不一致")
    if hQ_in != headQ or hK_in != headK:
        raise ValueError("headQ/headK 与输入形状不一致")
    if packDq != packDk:
        raise ValueError("q/k 的 packD 不一致")
    if headQ % headK != 0:
        raise ValueError("headQ 必须能被 headK 整除（以得到 Group）")

    B = B1
    Group = headQ // headK
    packD = packDq

    # 变形
    q = qhash_packed.reshape(B, headK, Group, packD)        # (B, headK, Group, packD)
    k = np.transpose(khash_packed, (0, 2, 1, 3))            # (B, headK, seqLen, packD)

    # 逐 uint16 XOR + popcount（对 16 bit）
    # 通过广播实现 (B, headK, Group, seqLen, packD)
    xor = np.bitwise_xor(q[:, :, :, None, :], k[:, :, None, :, :])

    # uint16 → popcount（0..16），并沿 packD 求和
    # LUT16[xor] 会产生 (B, headK, Group, seqLen, packD) 的 uint8 计数
    pop = _LUT16[xor]                      # uint8
    dist_per_group = pop.sum(axis=-1, dtype=np.int32)          # (B, headK, Group, seqLen)

    # 沿 Group 维累加 → (B, headK, seqLen)
    dist_sum = dist_per_group.sum(axis=2, dtype=np.int32)      # (B, headK, seqLen)

    # 扩成 (B, headK, 1, seqLen)
    dist = dist_sum[:, :, None, :]

    # chunk 压缩（在 seqLen 维，即最后一维）
    dist_chunk, ChunkS = _chunk_reduce_lastdim(dist, chunk_size=chunk_size, reduce=chunk_reduce)
    print(f"{dist_chunk}")

    # 取最小 topK（因为 Hamming 距离越小越好）
    topk_vals, topk_idx = _topk_lastdim(dist_chunk, k=topK, largest=False)

    return {
        "dist": dist.astype(np.int32),                     # (B, headK, 1, seqLen)
        "dist_chunk": dist_chunk.astype(np.int32),         # (B, headK, 1, ChunkS)
        "topk_vals": topk_vals.astype(np.int32),           # (B, headK, 1, topK)
        "topk_idx": topk_idx.astype(np.int32),             # (B, headK, 1, topK) —— chunk 索引
    }

def save_bin(path: str, arr: np.ndarray) -> None:
    """裸写为 .bin（无 header）。"""
    arr.tofile(path)

# ---------- 最小可运行示例 ----------
if __name__ == "__main__":
    # 配置
    # batchSize = 1
    # seqLen = 1024
    # headQ = 1
    # headK = 1
    # hidDim = 128
    # topK = 4

    batchSize = int(sys.argv[1])
    seqLen = int(sys.argv[2])
    headQ = int(sys.argv[3])
    headK = int(sys.argv[4])
    hidDim = int(sys.argv[5])
    topK = int(sys.argv[6])
    
    Group = headQ // headK
    assert(headQ == Group * headK)
    assert(hidDim % 16 == 0)  # 为简化示例，要求 hidDim 是 8 的倍数

    chunk_size = 16
    bitorder = "big"   # 若你的打包是 big-endian bits，可改为 "big"

    # 随机生成“打包态”的输入：最后一维是 ceil(hidDim/8)
    packD = (hidDim + 15) // 16
    rng = np.random.default_rng(1)
    qhash_packed = rng.integers(0, 128, size=(batchSize, 1, headQ, packD), dtype=np.uint16)
    khash_packed = rng.integers(0, 128, size=(batchSize, seqLen, headK, packD), dtype=np.uint16)

    Group = headQ // headK
    packD = hidDim // 16

    qhash_packed = rng.integers(0, 128, size=(batchSize, 1, headQ, packD), dtype=np.uint16)
    khash_packed = rng.integers(0, 128, size=(batchSize, seqLen, headK, packD), dtype=np.uint16)

    input_save_dir = "./input"
    save_bin(os.path.join(input_save_dir, "input_qhash.bin"), qhash_packed)
    save_bin(os.path.join(input_save_dir, "input_khash.bin"), khash_packed)
    

    out = hamming_topk_from_uint16_packed(
        qhash_packed=qhash_packed,
        khash_packed=khash_packed,
        headQ=headQ, headK=headK,
        topK=topK, chunk_size=16, chunk_reduce="max"
    )

    output_save_dir = "./output"
    save_bin(os.path.join(output_save_dir, "golden_topk_value.bin"), out["topk_vals"].astype(np.int32))
    save_bin(os.path.join(output_save_dir, "golden_topk_index.bin"), out["topk_idx"].astype(np.int32))

    print(out["dist"].shape, out["dist_chunk"].shape, out["topk_idx"].shape)
    print(out["topk_idx"][0,0,0,:])   # (B, headK, 1, topK)
    print(out["topk_vals"][0,0,0,:])   # (B, headK, 1, topK)
    # (2, 2, 1, 64) (2, 2, 1, 4) (2, 2, 1, 3)

