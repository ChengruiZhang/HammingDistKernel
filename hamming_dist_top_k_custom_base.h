#include "kernel_operator.h"
#include "lib/matmul_intf.h"


namespace AscendC {
    struct TilingParam {
        // basic info
        uint32_t batchSize;
        uint32_t seqLen;
        uint32_t totalNum; // totalNum = batchSize * seqLen
        uint32_t groupNum; // headQ / headK
        uint32_t chunkSize; // topk compress block -- 这里会有非整数问题
        uint32_t chunkNum; // ceil(SeqLen/chunkSize) 
        uint32_t compressTopK; // need add assert TopK/chunkSize 
        
        // uint32_t tileLength; 
        // uint32_t tileNum; // tileNum = Len / tileLength / BUFFERNUM
        uint32_t bufferNum;

        // Core Offset
        uint32_t qHashCoreOffset;
        uint32_t kHashCoreOffset;
        uint32_t indexCoreOffset;

        // tiling info
        uint32_t hDimTilingLen;
        uint32_t hDimTilingNum; // contain tail
        uint32_t hDimTilingTailLen;

        uint32_t seqLenTilingLen;
        uint32_t seqLenTilingNum;
        uint32_t seqLenTilingTailLen;
    }
}