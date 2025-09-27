/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void topk_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *qHashDevice, uint8_t *kHashDevice, uint8_t *topKDevice, uint8_t *workspace, uint8_t *tiling);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void topk_custom_do(uint32_t blockDim, void *l2ctrl, void *stream, uint8_t *qHashDevice, uint8_t *kHashDevice, uint8_t *topKDevice, uint8_t *workspace, uint8_t *tiling);
#endif

constexpr uint16_t BLOCK_DIM = 1;
constexpr uint16_t OUTTER = 1;
constexpr uint16_t INNER = 2048;
constexpr uint16_t N = 2048;
constexpr uint16_t K = 128;
constexpr uint16_t KPAD = 128;
constexpr bool ISLARGE = false;
constexpr uint32_t TILINGDATA_SIZE = 33 + 60 + 6;
constexpr uint32_t WORKSPACE_SIZE = 16 * 2048 * 1024;

typedef uint16_t GM_qHash_type;
typedef uint16_t GM_kHash_type;
typedef uint32_t GM_idx_type;


uint8_t* GenerateTiling(uint32_t batchSize,
                        uint32_t seqLen,
                        uint32_t headQ,
                        uint32_t headK,
                        uint32_t hidDim,
                        uint32_t topK,
                        uint32_t bufferNum,
                        const char *socVersion);


int32_t main(int32_t argc, char *argv[])
{
    uint32_t batchSize, seqLen, headQ, headK, hidDim, topK;

    std::cout << "Kernel Start" << std::endl;
    // parse args
    batchSize = std::stoi(argv[1]);
    seqLen = std::stoi(argv[2]);
    headQ = std::stoi(argv[3]);
    headK = std::stoi(argv[4]);
    hidDim = std::stoi(argv[5]);
    topK = std::stoi(argv[6]);
    int32_t deviceId = std::stoi(argv[7]);
    
    uint32_t chunkSize = 16;
    uint32_t bufferNum = 1;
    uint32_t blockDim = 1;
    const char *socVersion = nullptr;

    fprintf(stderr, "Operator Input Params: batchSize=%d, seqLen=%d, headQ=%d, headK=%d, hidDim=%d, topK=%d, chunkSize=%d\n", batchSize, seqLen, headQ, headK, hidDim, topK, chunkSize);

    size_t qHashFileSize = batchSize * 1 * headQ * hidDim / sizeof(GM_qHash_type) * sizeof(GM_qHash_type); // uint16_t represent 16 bool
    size_t kHashFileSize = batchSize * seqLen * headK * hidDim / sizeof(GM_kHash_type) * sizeof(GM_kHash_type);
    auto topKPad = (topK + chunkSize - 1) / chunkSize * chunkSize;
    size_t indexFileSize = batchSize * headK * topKPad * sizeof(GM_idx_type);

    // size_t inputSize_srcGmValue = OUTTER * INNER * sizeof(float);
    // size_t inputSize_srcGmIndex = OUTTER * INNER * sizeof(uint32_t);
    // size_t inputSize_finishGm = INNER * sizeof(bool);
    // size_t outputSize_dstGmValue = OUTTER * KPAD * sizeof(float);
    // size_t outputSize_dstGmIndex = OUTTER * KPAD * sizeof(uint32_t);

    // uint16_t kGm = K;
    // uint16_t outter = OUTTER;
    // uint16_t inner = INNER;
    // uint16_t n = N;
    // bool isLargestGm = ISLARGE;

    size_t workspaceSize = WORKSPACE_SIZE;
    size_t tilingFileSize = TILINGDATA_SIZE * sizeof(uint32_t);

#ifdef ASCENDC_CPU_DEBUG


#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *qHashHost, *kHashHost, *topKHost;
    uint8_t *qHashDevice, *kHashDevice, *topKDevice;

    // uint8_t *srcGmValueHost, *srcGmIndexHost, *finishGmHost, *dstGmValueHost, *dstGmIndexHost;
    // uint8_t *srcGmValueDevice, *srcGmIndexDevice, *finishGmDevice, *dstGmValueDevice,  *dstGmIndexDevice;
    uint8_t *workspaceHost;
    uint8_t *workspaceDevice, *tilingDevice;
    
    // CHECK_ACL(aclrtMallocHost((void **)(&srcGmValueHost), inputSize_srcGmValue));
    // CHECK_ACL(aclrtMallocHost((void **)(&srcGmIndexHost), inputSize_srcGmIndex));
    // CHECK_ACL(aclrtMallocHost((void **)(&finishGmHost), inputSize_finishGm));
    // CHECK_ACL(aclrtMallocHost((void **)(&dstGmValueHost), outputSize_dstGmValue));
    // CHECK_ACL(aclrtMallocHost((void **)(&dstGmIndexHost), outputSize_dstGmIndex));
    CHECK_ACL(aclrtMallocHost((void **)(&qHashHost), qHashFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&kHashHost), kHashFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&topKHost), indexFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&workspaceHost), workspaceSize));

    // CHECK_ACL(aclrtMalloc((void **)&srcGmValueDevice, inputSize_srcGmValue, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMalloc((void **)&srcGmIndexDevice, inputSize_srcGmIndex, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMalloc((void **)&finishGmDevice, inputSize_finishGm, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMalloc((void **)&dstGmValueDevice, outputSize_dstGmValue, ACL_MEM_MALLOC_HUGE_FIRST));
    // CHECK_ACL(aclrtMalloc((void **)&dstGmIndexDevice, outputSize_dstGmIndex, ACL_MEM_MALLOC_HUGE_FIRST));

    CHECK_ACL(aclrtMalloc((void **)&qHashDevice, qHashFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&kHashDevice, kHashFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&topKDevice, indexFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_qhash.bin", qHashFileSize, qHashHost, qHashFileSize);
    ReadFile("./input/input_khash.bin", kHashFileSize, kHashHost, kHashFileSize);

    // ReadFile("../input/input_srcGmValue.bin", inputSize_srcGmValue, srcGmValueHost, inputSize_srcGmValue);
    // ReadFile("../input/input_srcGmIndex.bin", inputSize_srcGmIndex, srcGmIndexHost, inputSize_srcGmIndex);
    // ReadFile("../input/input_finishGm.bin", inputSize_finishGm, finishGmHost, inputSize_finishGm);

    std::cout << "test 1 " << std::endl;
    CHECK_ACL(aclrtMemcpy(workspaceDevice, workspaceSize, workspaceHost, workspaceSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, GenerateTiling(batchSize, seqLen, headQ, headK, hidDim, topK, bufferNum, socVersion),
        tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    std::cout << "test 2 " << std::endl;

    // CHECK_ACL(aclrtMemcpy(srcGmValueDevice, inputSize_srcGmValue, srcGmValueHost, inputSize_srcGmValue, 
    //     ACL_MEMCPY_HOST_TO_DEVICE));
    // CHECK_ACL(aclrtMemcpy(srcGmIndexDevice, inputSize_srcGmIndex, srcGmIndexHost, inputSize_srcGmIndex, 
    //     ACL_MEMCPY_HOST_TO_DEVICE));
    // CHECK_ACL(aclrtMemcpy(finishGmDevice, inputSize_finishGm, finishGmHost, inputSize_finishGm, 
    //     ACL_MEMCPY_HOST_TO_DEVICE));

    CHECK_ACL(aclrtMemcpy(qHashDevice, qHashFileSize, qHashHost, qHashFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(kHashDevice, kHashFileSize, kHashHost, kHashFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    
    std::cout << "test 3 " << std::endl;
    topk_custom_do(blockDim, nullptr, stream, qHashDevice, kHashDevice, topKDevice, workspaceDevice, tilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));
    
    // CHECK_ACL(aclrtMemcpy(dstGmValueHost, outputSize_dstGmValue, dstGmValueDevice, outputSize_dstGmValue, 
    //     ACL_MEMCPY_DEVICE_TO_HOST));
    // CHECK_ACL(aclrtMemcpy(dstGmIndexHost, outputSize_dstGmIndex, dstGmIndexDevice, outputSize_dstGmIndex, 
    //     ACL_MEMCPY_DEVICE_TO_HOST));

    // WriteFile("../output/output_dstGmValue.bin", dstGmValueHost, outputSize_dstGmValue);
    // WriteFile("../output/output_dstGmIndex.bin", dstGmIndexHost, outputSize_dstGmIndex);

    CHECK_ACL(aclrtMemcpy(topKHost, indexFileSize, topKDevice, indexFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_topk_idx.bin", topKHost, indexFileSize);

    // bool goldenResult = true;
    // goldenResult &= CompareResult(dstGmValueHost, outputSize_dstGmValue, "dstGmValue");
    // goldenResult &= CompareResult(dstGmIndexHost, outputSize_dstGmIndex, "dstGmIndex");
    // if (goldenResult) {
    //     printf("test pass!\n");
    // } else {
    //     printf("test failed!\n");
    // }

    // CHECK_ACL(aclrtFree(srcGmValueDevice));
    // CHECK_ACL(aclrtFree(srcGmIndexDevice));
    // CHECK_ACL(aclrtFree(finishGmDevice));
    // CHECK_ACL(aclrtFree(dstGmValueDevice));
    // CHECK_ACL(aclrtFree(dstGmIndexDevice));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtFree(tilingDevice));

    // CHECK_ACL(aclrtFreeHost(srcGmValueHost));
    // CHECK_ACL(aclrtFreeHost(srcGmIndexHost));
    // CHECK_ACL(aclrtFreeHost(finishGmHost));
    // CHECK_ACL(aclrtFreeHost(dstGmValueHost));
    // CHECK_ACL(aclrtFreeHost(dstGmIndexHost));
    CHECK_ACL(aclrtFreeHost(workspaceHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
