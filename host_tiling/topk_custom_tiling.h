/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXAMPLES_SORT_TOPK_CUSTOM_TILING_H
#define EXAMPLES_SORT_TOPK_CUSTOM_TILING_H
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TopKCustomTilingData)
    
    TILING_DATA_FIELD_DEF(uint32_t, k); 
    TILING_DATA_FIELD_DEF(uint32_t, outter); 
    TILING_DATA_FIELD_DEF(uint32_t, inner); 
    TILING_DATA_FIELD_DEF(uint32_t, n); 
    TILING_DATA_FIELD_DEF(uint32_t, minsize); 
    TILING_DATA_FIELD_DEF(bool, isLargest);

    TILING_DATA_FIELD_DEF(uint32_t, batchSize);
    TILING_DATA_FIELD_DEF(uint32_t, seqLen);
    TILING_DATA_FIELD_DEF(uint32_t, seqLenPad);
    TILING_DATA_FIELD_DEF(uint32_t, seqBlock);

    TILING_DATA_FIELD_DEF(uint32_t, topK);
    TILING_DATA_FIELD_DEF(uint32_t, topKCompressed);
    TILING_DATA_FIELD_DEF(uint32_t, topKComprssedPad); // 注意：按你的字段名拼写

    TILING_DATA_FIELD_DEF(uint32_t, hidDim);
    TILING_DATA_FIELD_DEF(uint32_t, hidDimCompressNum);
    TILING_DATA_FIELD_DEF(uint32_t, hidDimCompressPadNum);
    TILING_DATA_FIELD_DEF(uint32_t, hidDimCompressAddNum);

    TILING_DATA_FIELD_DEF(uint32_t, totalNum);
    TILING_DATA_FIELD_DEF(uint32_t, groupNum);
    TILING_DATA_FIELD_DEF(uint32_t, bufferNum);

    TILING_DATA_FIELD_DEF(uint32_t, scalarSize);

    // *********** Core Offset -- per core -- for GM ***********
    TILING_DATA_FIELD_DEF(uint32_t, qHashCoreOffset);
    TILING_DATA_FIELD_DEF(uint32_t, kHashCoreOffset);
    TILING_DATA_FIELD_DEF(uint32_t, indexCoreOffset);
    TILING_DATA_FIELD_DEF(uint32_t, qHashCoreOffsetBlock);
    TILING_DATA_FIELD_DEF(uint32_t, kHashCoreOffsetBlock);
    TILING_DATA_FIELD_DEF(uint32_t, indexCoreOffsetBlock);

    // *********** tiling info -- elements ***********
    TILING_DATA_FIELD_DEF(uint32_t, seqLenTilingLen);
    TILING_DATA_FIELD_DEF(uint32_t, seqLenTilingNum);
    TILING_DATA_FIELD_DEF(uint32_t, seqLenTilingTailLen);
    TILING_DATA_FIELD_DEF(uint32_t, seqLenBlockNum);

    // Size info -- elements, not byte
    TILING_DATA_FIELD_DEF(uint32_t, qHashTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, qHashSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, kHashTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, kHashSingleTilingSize);

    TILING_DATA_FIELD_DEF(uint32_t, tmpWorkSpaceSize);

    // XOR / rightshift / reduceSum workspace
    TILING_DATA_FIELD_DEF(uint32_t, reduceSumWorkSpaceSize);

    TILING_DATA_FIELD_DEF(uint32_t, hammingXORTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingXORSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingRightTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingRightSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingCastTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingCastSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingLastRowTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingLastRowSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingSumTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingSumSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingCumTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingCumSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingReduceTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingReduceSingleTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingResultTilingSize);
    TILING_DATA_FIELD_DEF(uint32_t, hammingResultSingleTilingSize);

    TILING_DATA_FIELD_DEF(uint32_t, resultSize);
    TILING_DATA_FIELD_DEF(uint32_t, resultSingleSize);

    TILING_DATA_FIELD_DEF(uint32_t, resultChunkSize);
    TILING_DATA_FIELD_DEF(uint32_t, resultChunkSingleSize);

    // *********** topK ***********
    TILING_DATA_FIELD_DEF(uint32_t, chunkSize);
    TILING_DATA_FIELD_DEF(uint32_t, chunkRepeat);
    TILING_DATA_FIELD_DEF(uint32_t, chunkTailMask);
    TILING_DATA_FIELD_DEF(uint32_t, chunkMode);
    TILING_DATA_FIELD_DEF(uint32_t, chunkTopKNum);

    TILING_DATA_FIELD_DEF(uint32_t, indexChunkSize);
    TILING_DATA_FIELD_DEF(uint32_t, indexChunkSingleSize);
    TILING_DATA_FIELD_DEF(uint32_t, topKChunkSize);
    TILING_DATA_FIELD_DEF(uint32_t, topKChunkSingleSize);

    TILING_DATA_FIELD_DEF_STRUCT(TopkTiling, topKTilingData);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(TopkCustom, TopKCustomTilingData)
} // namespace optiling

static inline uint32_t ceil_div_u32(uint32_t x, uint32_t y) {
    return (x + y - 1) / y;
}
static inline uint32_t pad_to_u32(uint32_t x, uint32_t multiple) {
    return ceil_div_u32(x, multiple) * multiple;
}
void ComputeTiling(uint32_t batchSize,
                    uint32_t seqLen,
                    uint32_t headQ,
                    uint32_t headK,
                    uint32_t hidDim,
                    uint32_t topK,
                    uint32_t bufferNum,
                    optiling::TopKCustomTilingData &tiling)
{
    // 常量（与 Python 一致）
    const uint32_t dataBlockSize   = 16;
    const uint32_t chunkSize       = 16;
    const uint32_t seqLenTilingLen = 512;

    // 基本参数
    tiling.set_batchSize(batchSize);
    tiling.set_seqLen(seqLen);
    tiling.set_topK(topK);
    tiling.set_hidDim(hidDim);
    tiling.set_bufferNum(bufferNum);

    uint32_t groupNum = headQ / headK;
    assert((headQ % headK) == 0 && "HeadQ must be divisible by HeadK");
    tiling.set_groupNum(groupNum);
    tiling.set_totalNum(batchSize * headK);

    // padding / block 相关
    uint32_t seqLenPad = pad_to_u32(seqLen, dataBlockSize);
    uint32_t seqBlock  = ceil_div_u32(seqLen, dataBlockSize);
    tiling.set_seqLenPad(seqLenPad);
    tiling.set_seqBlock(seqBlock);

    uint32_t topKCompressed   = ceil_div_u32(topK, chunkSize);
    uint32_t topKComprssedPad = pad_to_u32(topKCompressed, dataBlockSize);
    tiling.set_topKCompressed(topKCompressed);
    tiling.set_topKComprssedPad(topKComprssedPad);

    uint32_t hidDimCompressNum    = ceil_div_u32(hidDim, dataBlockSize);
    uint32_t hidDimCompressPadNum = pad_to_u32(hidDimCompressNum, dataBlockSize);
    uint32_t hidDimCompressAddNum = hidDimCompressPadNum - hidDimCompressNum;
    tiling.set_hidDimCompressNum(hidDimCompressNum);
    tiling.set_hidDimCompressPadNum(hidDimCompressPadNum);
    tiling.set_hidDimCompressAddNum(hidDimCompressAddNum);

    tiling.set_scalarSize(64u * groupNum);

    // Core offsets（按元素计）
    tiling.set_qHashCoreOffset(groupNum * hidDimCompressNum);
    tiling.set_kHashCoreOffset(seqLen * hidDimCompressNum);
    tiling.set_indexCoreOffset(topKComprssedPad);

    tiling.set_qHashCoreOffsetBlock(tiling.get_qHashCoreOffset() / dataBlockSize);
    tiling.set_kHashCoreOffsetBlock(tiling.get_kHashCoreOffset() / dataBlockSize);
    tiling.set_indexCoreOffsetBlock(tiling.get_indexCoreOffset() / dataBlockSize);

    // tiling
    tiling.set_seqLenTilingLen(seqLenTilingLen);
    tiling.set_seqLenTilingNum(ceil_div_u32(seqLenPad, seqLenTilingLen));
    tiling.set_seqLenTilingTailLen((seqLen % seqLenTilingLen) ? (seqLen % seqLenTilingLen) : seqLenTilingLen);
    tiling.set_seqLenBlockNum(ceil_div_u32(seqLenTilingLen, dataBlockSize));

    // q/k Hash tiling sizes
    tiling.set_qHashTilingSize(groupNum * hidDimCompressPadNum * bufferNum);
    tiling.set_qHashSingleTilingSize(groupNum * hidDimCompressPadNum);
    tiling.set_kHashTilingSize(seqLenTilingLen * hidDimCompressPadNum * bufferNum);
    tiling.set_kHashSingleTilingSize(seqLenTilingLen * hidDimCompressPadNum);

    tiling.set_tmpWorkSpaceSize(groupNum * hidDimCompressPadNum * bufferNum);
    tiling.set_reduceSumWorkSpaceSize(512); // 覆盖 Python 取值

    // hamming 相关 sizes
    tiling.set_hammingXORTilingSize(groupNum * hidDimCompressPadNum * bufferNum);
    tiling.set_hammingXORSingleTilingSize(groupNum * hidDimCompressPadNum);
    tiling.set_hammingRightTilingSize(groupNum * seqLenTilingLen * bufferNum);
    tiling.set_hammingRightSingleTilingSize(groupNum * seqLenTilingLen);
    tiling.set_hammingCastTilingSize(groupNum * seqLenTilingLen * bufferNum);
    tiling.set_hammingCastSingleTilingSize(groupNum * seqLenTilingLen);
    tiling.set_hammingLastRowTilingSize(hidDimCompressPadNum * bufferNum);
    tiling.set_hammingLastRowSingleTilingSize(hidDimCompressPadNum);
    tiling.set_hammingSumTilingSize(hidDimCompressPadNum * seqLenTilingLen * bufferNum);
    tiling.set_hammingSumSingleTilingSize(hidDimCompressPadNum * seqLenTilingLen);
    tiling.set_hammingCumTilingSize(groupNum * seqLenTilingLen * bufferNum);
    tiling.set_hammingCumSingleTilingSize(groupNum * seqLenTilingLen);
    tiling.set_hammingReduceTilingSize(seqLenTilingLen * bufferNum);
    tiling.set_hammingReduceSingleTilingSize(seqLenTilingLen);
    tiling.set_hammingResultTilingSize(groupNum * seqLenTilingLen * bufferNum);
    tiling.set_hammingResultSingleTilingSize(groupNum * seqLenTilingLen);

    // result 区
    tiling.set_resultSize(seqLenPad * bufferNum);
    tiling.set_resultSingleSize(seqLenPad);
    uint32_t resultChunkSingleSize = ceil_div_u32(seqLenPad, chunkSize);
    tiling.set_resultChunkSingleSize(resultChunkSingleSize);
    tiling.set_resultChunkSize(resultChunkSingleSize * bufferNum);

    // topK 区
    tiling.set_chunkSize(chunkSize);
    tiling.set_chunkRepeat(ceil_div_u32(seqLen, chunkSize));
    {
        uint32_t rem = (seqLen % chunkSize);
        tiling.set_chunkTailMask((rem != 0) ? (static_cast<uint32_t>(1u << rem) - 1u) : 0xFFFFu);
    }
    tiling.set_chunkMode(0);
    tiling.set_chunkTopKNum(ceil_div_u32(topK, chunkSize));

    uint32_t indexChunkSingleSize = seqLenPad / chunkSize;
    tiling.set_indexChunkSingleSize(indexChunkSingleSize);
    tiling.set_indexChunkSize(indexChunkSingleSize * bufferNum);
    tiling.set_topKChunkSingleSize(topKComprssedPad);
    tiling.set_topKChunkSize(topKComprssedPad * bufferNum);
}

#endif // EXAMPLES_SORT_TOPK_CUSTOM_TILING_H
