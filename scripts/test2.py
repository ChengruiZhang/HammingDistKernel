import numpy as np

# 输入张量
arr = np.array([8,2,3,8,4,0,2,6,7,3,0,3,5,7,1,8,3,8,7,5,4,4,1,7,2,4,3,6,5,8,8,0,0,4,5,8,7
,2,0,7,1,4,3,6,9,7,4,2,6,4,1,1,6,9,4,9,1,4,5,8,3,9,7,2], dtype=np.uint8)


# 参考向量
ref = np.array([5,9,9,4,7,4,6,5], dtype=np.uint8)

# 每 8 个分块
blocks = arr.reshape(-1, 8)

# 计算 XOR + popcount
def popcount(x: int) -> int:
    return bin(x).count("1")

result = np.array([
    [popcount(v) for v in np.bitwise_xor(block, ref)]
    for block in blocks
])

print(result)
