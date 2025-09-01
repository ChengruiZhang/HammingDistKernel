import struct
import sys
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class HammingTilingData:
    # *********** basic info ***********
    batchSize: int = 0
    seqLen: int = 0
    seqLenPad: int = 0 # (seqLen + 16 - 1) // 16 * 16
    seqBlock: int = 0 # seqLenPad // 16

    topK: int = 0
    topKCompressed: int = 0
    topKComprssedPad: int = 0

    hidDim: int = 0
    hidDimCompressNum: int = 0
    hidDimCompressPadNum: int = 0
    hidDimCompressAddNum: int = 0

    totalNum: int = 0
    groupNum: int = 0
    bufferNum: int = 0

    scalarSize: int = 0

    # *********** Core Offset ***********
    qHashCoreOffset: int = 0
    kHashCoreOffset: int = 0
    indexCoreOffset: int = 0
    qHashCoreOffsetBlock: int = 0
    kHashCoreOffsetBlock: int = 0
    indexCoreOffsetBlock: int = 0

    # *********** tiling info ***********
    seqLenTilingLen: int = 0
    seqLenTilingNum: int = 0
    seqLenTilingTailLen: int = 0
    seqLenBlockNum: int = 0

    qHashTilingSize: int = 0
    qHashSingleTilingSize: int = 0
    kHashTilingSize: int = 0
    kHashSingleTilingSize: int = 0

    tmpWorkSpaceSize: int = 0
    reduceSumWorkSpaceSize: int = 512

    hammingXORTilingSize: int = 0
    hammingXORSingleTilingSize: int = 0
    hammingRightTilingSize: int = 0
    hammingRightSingleTilingSize: int = 0
    hammingCastTilingSize: int = 0
    hammingCastSingleTilingSize: int = 0
    hammingLastRowTilingSize: int = 0
    hammingLastRowSingleTilingSize: int = 0
    hammingSumTilingSize: int = 0
    hammingSumSingleTilingSize: int = 0
    hammingCumTilingSize: int = 0
    hammingCumSingleTilingSize: int = 0
    hammingReduceTilingSize: int = 0
    hammingReduceSingleTilingSize: int = 0
    hammingResultTilingSize: int = 0
    hammingResultSingleTilingSize: int = 0

    resultSize: int = 0
    resultSingleSize: int = 0

    resultChunkSize: int = 0
    resultChunkSingleSize: int = 0

    # *********** topK ***********
    chunkSize: int = 16
    chunkRepeat: int = 0
    chunkTailMask: int = 0
    chunkMode: int = 0
    chunkTopKNum: int = 0

    indexChunkSize: int = 0
    indexChunkSingleSize: int = 0
    topKChunkSize: int = 0
    topKChunkSingleSize: int = 0


    def to_bin(self, filename: str):
        """保存为 .bin 文件"""
        values = list(asdict(self).values())
        arr = np.array(values, dtype=np.uint32)
        arr.tofile(filename)

    @classmethod
    def from_bin(cls, filename: str):
        """从 .bin 文件读取"""
        arr = np.fromfile(filename, dtype=np.uint32)
        fields = list(cls.__dataclass_fields__.keys())
        kwargs = {k: int(v) for k, v in zip(fields, arr)}
        return cls(**kwargs)

if __name__ == "__main__":

    # if len(sys.argv) != 3:
    #     print("Usage: python gen_data_tmp.py <output_bin_file_name>")
    #     sys.exit(1)

    # input_txt_file = sys.argv[1]
    # output_bin_file = sys.argv[1]
    batchSize = int(sys.argv[1])
    seqLen = int(sys.argv[2])
    HeadQ = int(sys.argv[3])
    HeadK = int(sys.argv[4])
    hidDim = int(sys.argv[5])
    topK = int(sys.argv[6])
    bufferNum = int(sys.argv[7])

    groupNum = HeadQ // HeadK
    assert(HeadQ % HeadK == 0)

    UB_SIZE = 240 * 1024

    chunkSize = 16
    dataBlockSize = 16

    seqLenPad = (seqLen + 16 - 1) // 16 * 16
    seqBlock = (seqLen + 16 - 1) // 16

    topKCompressed = (topK + chunkSize - 1) // chunkSize
    topKComprssedPad = ((topK + chunkSize - 1) // chunkSize + 16 - 1) // 16 * 16

    hidDimCompressNum = (hidDim + dataBlockSize - 1) // dataBlockSize
    hidDimCompressPadNum = (hidDimCompressNum + dataBlockSize - 1) // dataBlockSize * dataBlockSize
    hidDimCompressAddNum = hidDimCompressPadNum - hidDimCompressNum

    totalNum = batchSize * groupNum
    scalarSize = 64 * groupNum

    qHashCoreOffset = groupNum * hidDimCompressNum
    kHashCoreOffset = seqLen * hidDimCompressNum
    indexCoreOffset = topKComprssedPad

    qHashCoreOffsetBlock = qHashCoreOffset // dataBlockSize
    kHashCoreOffsetBlock = kHashCoreOffset // dataBlockSize
    indexCoreOffsetBlock = indexCoreOffset // dataBlockSize

    seqLenTilingLen = 1024
    seqLenTilingNum = seqLenPad // seqLenTilingLen
    seqLenTilingTailLen = seqLenPad % seqLenTilingLen
    seqLenBlockNum = (seqLenTilingLen + dataBlockSize - 1) // dataBlockSize

    qHashTilingSize = groupNum * hidDimCompressPadNum * bufferNum
    qHashSingleTilingSize = groupNum * hidDimCompressPadNum
    kHashTilingSize = seqLenTilingLen * hidDimCompressPadNum * bufferNum
    kHashSingleTilingSize = seqLenTilingLen * hidDimCompressPadNum

    tmpWorkSpaceSize = groupNum * hidDimCompressPadNum * bufferNum
    reduceSumWorkSpaceSize = 512

    hammingXORTilingSize = groupNum * hidDimCompressPadNum * bufferNum
    hammingXORSingleTilingSize = groupNum * hidDimCompressPadNum
    hammingRightTilingSize = groupNum * seqLenTilingLen * bufferNum
    hammingRightSingleTilingSize = groupNum * seqLenTilingLen
    hammingCastTilingSize = groupNum * seqLenTilingLen * bufferNum
    hammingCastSingleTilingSize = groupNum * seqLenTilingLen
    hammingLastRowTilingSize = hidDimCompressPadNum * 1 * bufferNum
    hammingLastRowSingleTilingSize = hidDimCompressPadNum * 1
    hammingSumTilingSize = hidDimCompressPadNum * seqLenTilingLen * bufferNum
    hammingSumSingleTilingSize = hidDimCompressPadNum * seqLenTilingLen
    hammingCumTilingSize = groupNum * seqLenTilingLen * bufferNum
    hammingCumSingleTilingSize = groupNum * seqLenTilingLen
    hammingReduceTilingSize = seqLenTilingLen * bufferNum
    hammingReduceSingleTilingSize = seqLenTilingLen
    hammingResultTilingSize = groupNum * seqLenTilingLen * bufferNum
    hammingResultSingleTilingSize = groupNum * seqLenTilingLen

    resultSize = seqLenPad * bufferNum
    resultSingleSize = seqLenPad
    resultChunkSize = (seqLenPad + chunkSize - 1) // chunkSize * bufferNum
    resultChunkSingleSize = (seqLenPad + chunkSize - 1) // chunkSize

    # TBD
    chunkRepeat = (seqLen + chunkSize - 1) // chunkSize
    chunkTailMask = (1 << (seqLen % chunkSize)) - 1 if (seqLen % chunkSize) != 0 else 0xFFFF
    chunkMode = 0 if (seqLen % chunkSize) == 0 else 1
    chunkTopKNum = (topK + chunkSize - 1) // chunkSize

    indexChunkSize = seqLenPad // chunkSize * bufferNum
    indexChunkSingleSize = seqLenPad // chunkSize
    topKChunkSize = topKComprssedPad * bufferNum
    topKChunkSingleSize = topKComprssedPad

    totalBytes = (hammingXORTilingSize + hammingRightTilingSize + qHashTilingSize + kHashTilingSize + scalarSize) * 2 + (hammingSumTilingSize + hammingCumTilingSize + hammingReduceTilingSize + hammingCastTilingSize + hammingLastRowTilingSize + tmpWorkSpaceSize + reduceSumWorkSpaceSize + resultSize + resultChunkSize) * 2 + (indexChunkSize + topKChunkSize) * 4
    print(totalBytes)
    assert(totalBytes < UB_SIZE)

    data = {
        "batchSize": batchSize,
        "seqLen": seqLen,
        "seqLenPad": seqLenPad,
        "seqBlock": seqBlock,

        "topK": topK,
        "topKCompressed": topKCompressed,
        "topKComprssedPad": topKComprssedPad,

        "hidDim": hidDim,
        "hidDimCompressNum": hidDimCompressNum,
        "hidDimCompressPadNum": hidDimCompressPadNum,
        "hidDimCompressAddNum": hidDimCompressAddNum,

        "totalNum": totalNum,
        "groupNum": groupNum,
        "bufferNum": bufferNum,

        "scalarSize": scalarSize,

        # *********** Core Offset ***********
        "qHashCoreOffset": qHashCoreOffset,
        "kHashCoreOffset": kHashCoreOffset,
        "indexCoreOffset": indexCoreOffset,
        "qHashCoreOffsetBlock": qHashCoreOffsetBlock,
        "kHashCoreOffsetBlock": kHashCoreOffsetBlock,
        "indexCoreOffsetBlock": indexCoreOffsetBlock,

        # *********** tiling info ***********
        "seqLenTilingLen": seqLenTilingLen,
        "seqLenTilingNum": seqLenTilingNum,
        "seqLenTilingTailLen": seqLenTilingTailLen,
        "seqLenBlockNum": seqLenBlockNum,
        
        "qHashTilingSize": qHashTilingSize,
        "qHashSingleTilingSize": qHashSingleTilingSize,
        "kHashTilingSize": kHashTilingSize,
        "kHashSingleTilingSize": kHashSingleTilingSize,
        
        "tmpWorkSpaceSize": tmpWorkSpaceSize,
        "reduceSumWorkSpaceSize": reduceSumWorkSpaceSize,

        "hammingXORTilingSize": hammingXORTilingSize,
        "hammingXORSingleTilingSize": hammingXORSingleTilingSize,
        "hammingRightTilingSize": hammingRightTilingSize,
        "hammingRightSingleTilingSize": hammingRightSingleTilingSize,
        "hammingCastTilingSize": hammingCastTilingSize,
        "hammingCastSingleTilingSize": hammingCastSingleTilingSize,
        "hammingLastRowTilingSize": hammingLastRowTilingSize,
        "hammingLastRowSingleTilingSize": hammingLastRowSingleTilingSize,
        "hammingSumTilingSize": hammingSumTilingSize,
        "hammingSumSingleTilingSize": hammingSumSingleTilingSize,
        "hammingCumTilingSize": hammingCumTilingSize,
        "hammingCumSingleTilingSize": hammingCumSingleTilingSize,
        "hammingReduceTilingSize": hammingReduceTilingSize,
        "hammingReduceSingleTilingSize": hammingReduceSingleTilingSize,
        "hammingResultTilingSize": hammingResultTilingSize,
        "hammingResultSingleTilingSize": hammingResultSingleTilingSize,

        "resultSize": resultSize,
        "resultSingleSize": resultSingleSize,
        "resultChunkSize": resultChunkSize,
        "resultChunkSingleSize": resultChunkSingleSize,

        # *********** topK ***********
        "chunkSize": chunkSize,
        "chunkRepeat": chunkRepeat,
        "chunkTailMask": chunkTailMask,
        "chunkMode": chunkMode,
        "chunkTopKNum": chunkTopKNum,
        "indexChunkSize": indexChunkSize,
        "indexChunkSingleSize": indexChunkSingleSize,
        "topKChunkSize": topKChunkSize,
        "topKChunkSingleSize": topKChunkSingleSize,
    }


    
    hamming_data = HammingTilingData(**data)
    hamming_data.to_bin("./input/input_tiling.bin")
    print(f"Data saved to ./input/input_tiling.bin")