/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUuDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include "tiling/tiling_api.h"
#include "../host_tiling/topk_custom_tiling.h"

uint8_t *GetTilingBuf(optiling::TopKCustomTilingData *tilingData) {

  uint32_t tilingSize = sizeof(optiling::TopKCustomTilingData);
  uint8_t *buf = (uint8_t *)malloc(tilingSize);

  tilingData->SaveToBuffer(buf, tilingSize);

  return buf;

}

uint8_t* GenerateTiling(uint32_t batchSize,
                        uint32_t seqLen,
                        uint32_t headQ,
                        uint32_t headK,
                        uint32_t hidDim,
                        uint32_t topK,
                        uint32_t bufferNum,
                        const char *socVersion)
{

    optiling::TopKCustomTilingData tiling;
    ComputeTiling(batchSize, seqLen, headQ, headK, hidDim, topK, bufferNum, tiling);

    uint32_t maxsize = 0;
    uint32_t minsize = 0;
    uint32_t dtypesize = 4;  // float类型
    
    uint32_t chunkSize = 16;
    uint32_t k = topK;
    uint32_t outter = 1;
    uint32_t inner = (seqLen + chunkSize - 1) / chunkSize * chunkSize; // 向上取整到chunkSize的整数倍
    uint32_t n = inner;
    bool isLargest = false;

    platform_ascendc::PlatformAscendC* ascendcPlatform;
    if (socVersion != nullptr) {
        ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    } else {
        ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    }
    const platform_ascendc::PlatformAscendC& ascendCPlatform = *ascendcPlatform;
    AscendC::TopKTilingFunc(ascendCPlatform, inner, outter, topK, dtypesize,
        false, AscendC::TopKMode::TOPK_NORMAL, isLargest, tiling.topKTilingData);
    AscendC::GetTopKMaxMinTmpSize(ascendCPlatform, inner, outter, false,
        false, AscendC::TopKMode::TOPK_NORMAL, isLargest, dtypesize, maxsize, minsize);
    tiling.set_minsize(minsize);
    tiling.set_k(k);
    tiling.set_outter(outter);
    tiling.set_inner(inner);
    tiling.set_n(n);
    tiling.set_isLargest(isLargest);

    return GetTilingBuf(&tiling);
}