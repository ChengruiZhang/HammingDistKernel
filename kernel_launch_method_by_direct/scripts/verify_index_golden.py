import numpy as np
import sys


def compare_bin_files_blocked(file1: str, file2: str, dtype: str, N: int, max_mismatch_report: int = 10) -> bool:
    """
    按块验证两个 .bin 文件：
      - 文件1：每轮读取 ceil(N/8)*8 个元素，只比较前 N 个，其余作为对齐填充跳过
      - 文件2：每轮读取恰好 N 个元素，与文件1的前 N 个比较
    直到任一文件数据不足为止；若在最后一轮两边都读不到满额数据，则停止比较并报告。

    返回 True 表示所有比较块均一致，否则 False。
    """
    if N <= 0:
        raise ValueError("N 必须 > 0")

    dt = np.dtype(dtype)
    itemsize = dt.itemsize

    pad = (8 - (N % 8)) % 8          # 需要跳过的对齐数量（元素数）
    read_len1 = N + pad              # 文件1每轮实际读取的元素个数（8 对齐）
    read_bytes1 = read_len1 * itemsize
    read_bytes2 = N * itemsize

    total_compared = 0
    total_mismatch = 0
    block_idx = 0

    with open(file1, "rb") as f1, open(file2, "rb") as f2:
        while True:
            # 读文件1：按对齐后的长度
            buf1 = f1.read(read_bytes1)
            if len(buf1) == 0:
                # 两边都刚好读完 → 成功结束
                break
            if len(buf1) < read_bytes1:
                print(f"⚠️ 文件1在块 {block_idx} 数据不足：期望 {read_bytes1} 字节，实际 {len(buf1)}，停止比较。")
                return False

            # 读文件2：恰好 N 个元素
            buf2 = f2.read(read_bytes2)
            if len(buf2) < read_bytes2:
                print(f"⚠️ 文件2在块 {block_idx} 数据不足：期望 {read_bytes2} 字节，实际 {len(buf2)}，停止比较。")
                return False

            # 解析为数组
            arr1_full = np.frombuffer(buf1, dtype=dt)           # 长度 read_len1
            arr1 = arr1_full[:N]                                 # 仅取前 N
            arr2 = np.frombuffer(buf2, dtype=dt)                 # 长度 N

            # 比较
            eq = (arr1 == arr2)
            if not np.all(eq):
                mismatch_idx = np.where(~eq)[0]
                total_mismatch += mismatch_idx.size
                print(f"❌ 块 {block_idx} 存在不一致：{mismatch_idx.size} 处不同（仅显示前 {max_mismatch_report} 处）")
                for i in mismatch_idx[:max_mismatch_report]:
                    print(f"  - 全局元素 #{total_compared + i}: file1={arr1[i]!r}, file2={arr2[i]!r}")

            total_compared += N
            block_idx += 1

    if total_mismatch == 0:
        print(f"✅ 完成：共比较 {total_compared} 个元素，全部一致。")
        return True
    else:
        print(f"❗完成：共比较 {total_compared} 个元素，发现 {total_mismatch} 处不一致。")
        return False


# ========== 示例 ==========
if __name__ == "__main__":
    # 假设有两个文件 a.bin 和 b.bin，每个存 int32
    fileA = sys.argv[1] if len(sys.argv) > 1 else "a.bin"
    fileB = sys.argv[2] if len(sys.argv) > 2 else "b.bin"
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    ok = compare_bin_files_blocked(fileA, fileB, dtype="int32", N=N)
