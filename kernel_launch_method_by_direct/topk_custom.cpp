/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include <iostream>
#include "../kernel_impl/topk_custom.h"

__aicore__ inline void CopyTiling(MyCustomKernel::VecTiling* tiling, GM_ADDR tilingGM)
{
    uint32_t* ptr = reinterpret_cast<uint32_t*>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t*>(tilingGM);

    for (uint32_t i = 0; i < sizeof(MyCustomKernel::VecTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

extern "C" __global__ __aicore__ void topk_custom(GM_ADDR qHash, GM_ADDR kHash, 
                                                  GM_ADDR topK, GM_ADDR workspace, GM_ADDR tiling)
{
    if ASCEND_IS_AIC {
        return;
    }

    AscendC::PRINTF("1\n");

    MyCustomKernel::VecTiling tilingData;
    CopyTiling(&tilingData, tiling);

    MyCustomKernel::KernelTopK<int16_t, half, int32_t> op(tilingData);
    
    // AscendC::PRINTF("%d\n", tilingData.indexChunkSingleSize);
    op.CopyTiling(tilingData);
    op.Init(qHash, kHash, topK, tilingData);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void topk_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *qHashDevice, uint8_t *kHashDevice, uint8_t *topKDevice, uint8_t *workspace, uint8_t *tiling)
{
    std::cout << "test" << std::endl;
    topk_custom<<<blockDim, l2ctrl, stream>>>(qHashDevice, kHashDevice, topKDevice, workspace, tiling);
}
#endif
