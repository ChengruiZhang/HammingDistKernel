#include "kernel_operator.h"
#include "lib/matmul_intf.h"


namespace AscendC {
    struct TilingParam {

        // *********** basic info *********** 
        uint32_t batchSize;
        uint32_t seqLen;
        uint32_t seqLenPad; // pad elements to 32 Bytes
        uint32_t seqBlock; // ceil(seqLenPad / 16) -- 每个datablock有16个元素

        uint32_t topK;
        uint32_t topKCompressed; // topK after compression
        uint32_t topKComprssedPad; // pad to 32 Byte

        uint32_t hidDim;
        uint32_t hidDimCompressNum; // bool -- int16/int8 压缩之后的实际elements大小
        uint32_t hidDimCompressPadNum; // pad to 32 Bytes -- elements大小
        uint32_t hidDimCompressAddNum; // pad增加的element个数
        
        uint32_t totalNum; // totalNum = batchSize * headK
        uint32_t groupNum; // headQ / headK
        uint32_t bufferNum;

        uint32_t scalarSize; // hamming Scalar -- 4 * 64 * Group = 256 * group bits = 32 * group bytes = 16 * group elements


        // *********** Core Offset -- per core -- for GM ***********
        uint32_t qHashCoreOffset; // G * hidDimCompressNum * sizeof(hashDataType)
        uint32_t kHashCoreOffset; // seqLen * hidDimCompressNum * sizeof(hashDataType)
        uint32_t indexCoreOffset; // topKComprssedPad * sizeof(indexDataType)
        uint32_t qHashCoreOffsetBlock; // 32 Byte = 1 block
        uint32_t kHashCoreOffsetBlock;
        uint32_t indexCoreOffsetBlock;

        // *********** tiling info -- 均为elements为单元 ***********
        uint32_t seqLenTilingLen; // 需要满足32 Byte 对齐
        uint32_t seqLenTilingNum;
        uint32_t seqLenTilingTailLen;
        uint32_t seqLenBlockNum; // (seqLenTilingLen + 15) / 16
        
        // Size info -- elements, not byte
        uint32_t qHashTilingSize; // G * bfn * hidDimCompressPadNum
        uint32_t qHashSingleTilingSize; // G * hidDimCompressPadNum
        uint32_t kHashTilingSize; // seqLenTilingLen * hidDimCompressPadNum * bfn
        uint32_t kHashSingleTilingSize; // seqLenTilingLen * hidDimCompressPadNum

        uint32_t tmpWorkSpaceSize; // 临时工作空间，用于存放kHash的复制值，保持和qHash一致的shape，32 Byte对齐 -- G * hidDimCompressPadNum * bfn

        // XOR rightshift 这些都是对T_SeqLen中的一个做的
        uint32_t reduceSumWorkSpaceSize = 512; // reduceSum的工作空间，32 Byte对齐

        uint32_t hammingXORTilingSize; // G * hidDimCompressPadNum * bfn
        uint32_t hammingXORSingleTilingSize; // G * hidDimCompressPadNum
        uint32_t hammingRightTilingSize; // G * hidDimCompressPadNum * bfn
        uint32_t hammingRightSingleTilingSize; // G * hidDimCompressPadNum
        uint32_t hammingCastTilingSize; // G * hidDimCompressPadNum * bfn
        uint32_t hammingCastSingleTilingSize; // G * hidDimCompressPadNum
        uint32_t hammingLastRowTilingSize; // 1 * hidDimCompressPadNum * bfn
        uint32_t hammingLastRowSingleTilingSize; // 1 * hidDimCompressPadNum
        uint32_t hammingSumTilingSize; // seqLenTilingLen * DATABLOCKLEN * bfn
        uint32_t hammingSumSingleTilingSize; // seqLenTilingLen * DATABLOCKLEN
        uint32_t hammingCumTilingSize; // seqLenTilingLen * DATABLOCKLEN * bfn
        uint32_t hammingCumSingleTilingSize; // seqLenTilingLen * DATABLOCKLEN
        uint32_t hammingReduceTilingSize; // seqLenTilingLen * DATABLOCKLEN * bfn
        uint32_t hammingReduceSingleTilingSize; // seqLenTilingLen * DATABLOCKLEN
        uint32_t hammingResultTilingSize; // seqLenTilingLen * bfn
        uint32_t hammingResultSingleTilingSize; // seqLenTilingLen
        
        uint32_t resultSize; // seqLenPad * bfn
        uint32_t resultSingleSize; // seqLenPad

        uint32_t resultChunkSize; // (seqLenPad + chunkSize - 1) / chunkSize * bfn
        uint32_t resultChunkSingleSize; // (seqLenPad + chunkSize - 1) / chunkSize

        // *********** topK *********** 
        uint32_t chunkSize; // topk compress block -- 这里会有非整数问题 -- 仅为16
        uint32_t chunkRepeat; // ceil(seqBlock / 8)
        uint32_t chunkTailMask; // (seqLen % (8 * 16)) -- 可能为0
        uint32_t chunkMode;  // max
        uint32_t chunkTopKNum; // need add assert TopK/chunkSize
        
        //  ((seqLenPad + 128 - 1) / 128 * 8 + 16 - 1) * 16 * bfn
        uint32_t indexChunkSize; // seqLenPad / chunkSize  
        uint32_t indexChunkSingleSize;
        uint32_t topKChunkSize; // (k + 16 - 1) * 16 * bfn
        uint32_t topKChunkSingleSize; // (k + 16 - 1) * 16
        
        // // useless
        
        // // 这里的hDim均为压缩后的
        // uint32_t hDimTilingLen;
        // uint32_t hDimTilingNum = 1; // 
        // uint32_t hDimTilingTailLen = 0;
        
        // uint32_t qHashCoreOffsetBlock; // 32 Byte = 1 block
        // uint32_t kHashCoreOffsetBlock;
        // uint32_t indexCoreOffsetBlock;
        
        // uint32_t hammingReduceTilingSize; // G * 16 (DATABLOCKLEN, 按0扩充至32 Byte) * bfn 
        // uint32_t hammingReduceSingleTilingSize; // G * 16 (DATABLOCKLEN) 

        // uint32_t chunkNum; // ceil(SeqLen / chunkSize)  -- 似乎无用
        // uint32_t chunkTail; // SeqLen % chunkSize  -- 似乎无用
    };
}