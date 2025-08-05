/**
 * @file hamming_dist_top_k_custom_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co. Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef Hamming_CUSTOM_TILING_H
#define Hamming_CUSTOM_TILING_H
#include <cstdint>

struct HammingTilingData {
    
    // basic info
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t hidDim;
    uint32_t hidDimCompress; // bool -- int16/int8
    uint32_t totalNum; // totalNum = batchSize * seqLen
    uint32_t groupNum; // headQ / headK
    uint32_t chunkSize; // topk compress block -- 这里会有非整数问题
    uint32_t chunkNum; // ceil(SeqLen/chunkSize) 
    uint32_t chunkTopKNum; // need add assert TopK/chunkSize 
    
    // Core Offset
    uint32_t qHashCoreOffset;
    uint32_t kHashCoreOffset;
    uint32_t indexCoreOffset;

    // tiling info
    uint32_t seqLenTilingLen;
    uint32_t seqLenTilingNum;
    uint32_t seqLenTilingTailLen;
    
    // 这里的hDim均为压缩后的
    uint32_t hDimTilingLen;
    uint32_t hDimTilingNum; // contain tail
    uint32_t hDimTilingTailLen;
    
    // Size info -- elements, not byte
    uint32_t bufferNum;
    uint32_t qHashTilingSize; // contain buffer num -- 一次性读完全部的，HDim切块为1 -- G*bfn*HDim
    uint32_t qHashSingleTilingSize; // G*HDim
    uint32_t kHashTilingSize; // contain buffer num -- T_SeqLen * HDim * bfn
    uint32_t kHashSingleTilingSize; // T_SeqLen * HDim
    // // 注意，这里的qHashGroup为整个hidden Dim的Size，不是分块之后的
    // uint32_t qHashGroupSize;  // contain buffer num  
    // uint32_t qHashGroupSingleSize;
    uint32_t indexChunkTilingSize;  // contain buffer num  
    uint32_t indexChunkSingleTilingSize;
    uint32_t hammingGroupTilingSize; // contain buffer num and tiling -- G * T_SeqLen * bfn
    uint32_t hammingGroupSingleTilingSize; // G * T_SeqLen    
    uint32_t hammingSumTilingSize; // contain buffer num and tiling -- T_SeqLen * bfn
    uint32_t hammingSumSingleTilingSize; // T_SeqLen
    uint32_t hammingChunkTilingSize; // ChunkNum * bfn
    uint32_t hammingChunkSingleTilingSize; // ChunkNum
    uint32_t topKChunkTilingSize; // chunkTopKNum * bfn
    uint32_t topkChunkSingleTilingSize; // chunkTopKNum

};
#endif