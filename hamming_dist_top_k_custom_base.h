#include "kernel_operator.h"
#include "lib/matmul_intf.h"


namespace AscendC {
    struct TilingParam {
        // basic info
        uint32_t batchSize;
        uint32_t seqLen;
        uint32_t seqLenPad; // pad elements to 32 Bytes
        uint32_t seqBlock; // ceil(seqLenPad / 16) -- 每个datablock有16个元素
        uint32_t reduceSumWorkSpace = 512; // reduceSum的工作空间，32 Byte对齐

        uint32_t hidDim;
        uint32_t hidDimCompressNum; // bool -- int16/int8 压缩之后的实际elements大小
        uint32_t hidDimCompressPadNum; // pad to 32 Bytes -- elements大小
        uint32_t hidDimCompressAddNum; // pad增加的element个数
        
        uint32_t totalNum; // totalNum = batchSize * headK
        uint32_t groupNum; // headQ / headK
        
        uint32_t chunkSize; // topk compress block -- 这里会有非整数问题
        uint32_t chunkRepeat; // ceil(seqBlock / 8)
        uint32_t chunkTailMask; // (seqLen % (8 * 16)) -- 可能为0
        uint32_t chunkNum; // ceil(SeqLen / chunkSize)  -- 似乎无用
        uint32_t chunkTail; // SeqLen % chunkSize  -- 似乎无用
        uint32_t chunkMode;  // max
        uint32_t chunkTopKNum; // need add assert TopK/chunkSize

        uint32_t scalarSize; // hamming Scalar -- 4 * 64 = 256 bits = 64 bytes = 32 elements

        // Core Offset -- per core
        uint32_t qHashCoreOffset; // G * hidDimCompressPadNum * sizeof(hashDataType)
        uint32_t kHashCoreOffset; // G * hidDimCompressPadNum * sizeof(hashDataType)
        uint32_t indexCoreOffset; 
        uint32_t qHashCoreOffsetBlock; // 32 Byte = 1 block
        uint32_t kHashCoreOffsetBlock;
        uint32_t indexCoreOffsetBlock;

        // tiling info -- elements
        uint32_t seqLenTilingLen;
        uint32_t seqLenTilingLenPad; // 在尾块中补齐，尾块前为正常
        uint32_t seqLenTilingNum;
        uint32_t seqLenTilingTailLen;
        uint32_t seqLenBlockNum; // seqLenTilingLenPad / 16
        
        // 这里的hDim均为压缩后的
        uint32_t hDimTilingLen;
        uint32_t hDimTilingNum; // contain tail
        uint32_t hDimTilingTailLen;
        
        // Size info -- elements, not byte
        uint32_t bufferNum;
        uint32_t qHashTilingSize; // contain buffer num -- 一次性读完全部的，HDim切块为1 -- G*bfn*hidDimCompressPadNum
        uint32_t qHashSingleTilingSize; // G*hidDimCompressPadNum
        uint32_t kHashTilingSize; // contain buffer num -- T_SeqLenPad * hidDimCompressPadNum * bfn
        uint32_t kHashSingleTilingSize; // T_SeqLenPad * hidDimCompressPadNum
        
        
        // hamming -- 2个临时空间足矣
        //     unsigned int popcount16(unsigned int x) {
        //     x = x - ((x >> 1) & 0x5555);               // 每2位
        //     x = (x & 0x3333) + ((x >> 2) & 0x3333);    // 每4位
        //     x = (x + (x >> 4)) & 0x0F0F;               // 每8位
        //     x = (x + (x >> 8)) & 0x001F;               // 合并到16位
        //     return x;
        // }

        // XOR rightshift 这些都是对T_SeqLen中的一个做的
        uint32_t hammingXORTilingSize; // contain buffer num and tiling -- G * hidDimCompressPadNum * bfn
        uint32_t hammingXORSingleTilingSize; // G * hidDimCompressPadNum
        uint32_t hammingRightTilingSize; // G * hidDimCompressPadNum * bfn
        uint32_t hammingRightSingleTilingSize; // G * hidDimCompressPadNum
        uint32_t hammingSumTilingSize; // contain buffer num and tiling -- T_SeqLenPad * DATABLOCKLEN * bfn
        uint32_t hammingSumSingleTilingSize; // T_SeqLenPad * DATABLOCKLEN
        uint32_t hammingReduceTilingSize; // G * 16 (DATABLOCKLEN, 按0扩充至32 Byte) * bfn 
        uint32_t hammingReduceSingleTilingSize; // G * 16 (DATABLOCKLEN) 
        uint32_t hammingResultTilingSize; // contain buffer num and tiling -- T_SeqLenPad * bfn
        uint32_t hammingResultSingleTilingSize; // T_SeqLenPad
        uint32_t hammingChunkTilingSize; // T_SeqLenPad / 16 * bfn -- block reduce 能够做维度缩减
        uint32_t hammingChunkSingleTilingSize; // T_SeqLenPad / 16
        
        uint32_t indexChunkTilingSize;  //  ((seqLenPad + 128 - 1) / 128 * 8 + 16 - 1) * 16 * bfn
        uint32_t indexChunkSingleTilingSize;
        uint32_t topKChunkTilingSize; // (k + 16 - 1) * 16 * bfn
        uint32_t topKChunkSingleTilingSize; // (k + 16 - 1) * 16

    };
}