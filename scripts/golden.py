import numpy as np
import sys

def count_bit_ones_uint16(x: np.ndarray) -> np.ndarray:
    return np.unpackbits(x.view(np.uint8), axis=-1).sum(axis=-1)

def reduce_hashq_groupwise(hashq, group):
    """
    hashq: [B, 1, Hq, D_Block], uint16
    group: int, e.g., 2 表示每两个 head 合并(此处对算法部分仍有问题)
    return: [B, 1, Hk, D_Block], uint16
    """
    B, _, Hq, D_Block = hashq.shape
    assert Hq % group == 0, "Hq must be divisible by group"
    Hk = Hq // group
    hashq_grouped = hashq.reshape(B, 1, Hk, group, D_Block)  # [B, 1, Hk, group, D_Block]
    reduced = np.sum(hashq_grouped, axis=3, dtype=np.uint32)  # 避免溢出
    return reduced.astype(np.uint16)

def compute_hamming_distance(hashq, hashk):
    """
    hashq: [B, 1, Hk, D_Block]
    hashk: [B, S, Hk, D_Block]
    return: [B, Hk, S]
    """
    B, _, Hk, D_Block = hashq.shape
    _, S, _, _ = hashk.shape
    q_exp = hashq[:, 0, :, None, :]        # [B, Hk, 1, D_Block]
    k_exp = hashk.transpose(0, 2, 1, 3)    # [B, Hk, S, D_Block]
    xor = np.bitwise_xor(q_exp, k_exp)    # [B, Hk, S, D_Block]
    hamming = count_bit_ones_uint16(xor)  # [B, Hk, S]
    return hamming

def get_topk_indices(hashq, hashk, topk, group):
    """
    hashq: [B, 1, Hq, D_Block]
    hashk: [B, S, Hk, D_Block]
    return: [B, Hk, topk]
    """
    B, _, Hq, D_Block = hashq.shape
    _, S, Hk, _ = hashk.shape
    assert Hq == Hk * group, "Hq must equal Hk * group"
    hashq_reduced = reduce_hashq_groupwise(hashq, group)   # [B, 1, Hk, D_Block]
    dists = compute_hamming_distance(hashq_reduced, hashk) # [B, Hk, S]
    idx = np.argsort(dists, axis=-1)[..., :topk]           # [B, Hk, topk]
    return idx, dists

def expand_ndarray_broadcast(arr, K):
    return arr[..., None] * K + np.arange(K)

if __name__ == "__main__":
    # 示例测试
    if len(sys.argv) < 7:
        print("Not enough input Num")
        sys.exit(1)

    B, S, Hq, Hk, D = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])  # => Hk = Hq // group = 4
    group = Hq // Hk
    assert(Hq % Hk == 0)
    hashGroup = 16 # bool 16 --> uint16
    D_Block = int(D / hashGroup)
    topk = int(sys.argv[6])

    np.random.seed(0)
    hashq = np.random.randint(0, 2**16, size=(B, 1, Hq, D_Block), dtype=np.uint16)
    hashq.tofile("./input/input_qhash.bin")
    hashk = np.random.randint(0, 2**16, size=(B, S, Hk, D_Block), dtype=np.uint16)
    hashk.tofile("./input/input_khash.bin")

    topk_idx, dists = get_topk_indices(hashq, hashk, topk, group)

    topk_idx.tofile("./output/output_topk_idx.bin")
    expanded = expand_ndarray_broadcast(topk_idx, hashGroup)
    topk_idx.tofile("./output/output_topk_idx_expand.bin")

    print("Top-k indices shape:", topk_idx.shape)  # [B, Hk, topk]
    print("Hamming distances:", dists)  # [B, Hk, S]
    print("Top-k indices:", topk_idx)
    print("Top-k indices expanded:", expanded)
