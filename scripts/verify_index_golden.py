import numpy as np
import sys

def compare_bin_files(file1: str, file2: str, dtype: str, N: int) -> bool:
    """
    验证两个 .bin 文件的前 N 个数是否一致。
    
    参数:
        file1 : 第一个文件路径
        file2 : 第二个文件路径
        dtype : 数据类型，例如 'int32', 'float32', 'uint8' 等
        N     : 比较前 N 个元素
    
    返回:
        True  : 前 N 个数完全一致
        False : 存在不同
    """
    # 读取前 N 个
    arr1 = np.fromfile(file1, dtype=dtype, count=N)
    arr2 = np.fromfile(file2, dtype=dtype, count=N)
    
    if arr1.shape[0] < N or arr2.shape[0] < N:
        print(f"⚠️ 文件数据不足 N={N} 个元素")
        return False

    # 比较
    equal = np.array_equal(arr1, arr2)
    if equal:
        print(f"✅ 前 {N} 个元素一致")
    else:
        print(f"❌ 前 {N} 个元素不一致")
        # 打印不同的位置和数值
        diff_idx = np.where(arr1 != arr2)[0]
        print("不同位置:", diff_idx)
        for i in diff_idx:
            print(f"  idx={i}: file1={arr1[i]}, file2={arr2[i]}")
    return equal


# ========== 示例 ==========
if __name__ == "__main__":
    # 假设有两个文件 a.bin 和 b.bin，每个存 int32
    fileA = sys.argv[1] if len(sys.argv) > 1 else "a.bin"
    fileB = sys.argv[2] if len(sys.argv) > 2 else "b.bin"
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    ok = compare_bin_files(fileA, fileB, dtype="int32", N=N)
