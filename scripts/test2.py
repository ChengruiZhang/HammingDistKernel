import numpy as np

def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    xor = np.bitwise_xor(a, b)
    return np.unpackbits(xor.view(np.uint8)).sum()

def blockwise_sum(a: np.ndarray, b: np.ndarray) -> int:
    xor = np.bitwise_xor(a, b)
    xor_uint16 = xor.view(np.uint16)
    return xor_uint16.sum()

# 示例：32-bit 输入
a = np.array([0b00000000, 0b00001111, 0b11110000, 0b10101010], dtype=np.uint8)
b = np.array([0b00001111, 0b00001111, 0b00001111, 0b01000000], dtype=np.uint8)

print("Hamming distance:", hamming_distance(a, b))
print("Blockwise sum:", blockwise_sum(a, b))
