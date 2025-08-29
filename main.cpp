/**
 * @file main.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "hamming_dist_top_k_custom_tiling.h"
#include "data_utils.h"
#include "common.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_hamming_dist_top_k_custom.h"
#include "tiling/platform/platform_ascendc.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void hamming_dist_top_k_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, HammingTilingData tiling);
#endif
extern void GenerateTilingData(uint8_t* tilingBuf, uint32_t blockDim);

/*
@brief Main function for Hamming distance top-k custom kernel with tiling support.
@param qhash: [B, 1, Hq, D], bool --> [B, 1, Hq, D], uint16_t -- 重要：为了保证右移计算的正确性，所有数据将会转化为uint16_t类型，可以按照其他类型输入，但会转化成uint16
@param khash: [B, S, Hk, D], bool --> [B, S, Hq, D]
@param topk: int32_t
@param topk_index: [B, Hk, topk], uint16_t
*/

int32_t main(int32_t argc, char *argv[])
{
    uint32_t batchSize, seqLen, headQ, headK, hidDim, topK, chunkSize;

    // fprintf(1);
    std::cout << 1 << std::endl;
    // parse args
    batchSize = std::stoi(argv[1]);
    seqLen = std::stoi(argv[2]);
    headQ = std::stoi(argv[3]);
    headK = std::stoi(argv[4]);
    hidDim = std::stoi(argv[5]);
    topK = std::stoi(argv[6]);
    int32_t deviceId = std::stoi(argv[7]);
    chunkSize = 16;

    fprintf(stderr, "Operator Input Params: batchSize=%d, seqLen=%d, headQ=%d, headK=%d, hidDim=%d, topK=%d, chunkSize=%d\n", 
            batchSize, seqLen, headQ, headK, hidDim, topK, chunkSize);

    assert(headQ % headK == 0 && "headQ must be divisible by headK");
    assert(hidDim % sizeof(GM_qHash_type) == 0 && "hidDim must be divisible by 16");

    size_t qHashFileSize = batchSize * 1 * headQ * hidDim / sizeof(GM_qHash_type) * sizeof(GM_qHash_type); // uint16_t represent 16 bool
    size_t kHashFileSize = batchSize * seqLen * headK * hidDim / sizeof(GM_kHash_type) * sizeof(GM_kHash_type);
    auto compressedTopK = (topK + chunkSize - 1) / chunkSize;
    size_t indexFileSize = batchSize * headK * compressedTopK * sizeof(GM_idx_type);

    // 待定
    constexpr uint32_t BLOCK_DIM = 1;
    uint8_t *tiling = nullptr;
    // constexpr uint32_t DATA_TYPE_SIZE[] = {2, 2, 4, 1, 2, 4};
    size_t tilingSize = 60 * sizeof(uint32_t);

#ifdef ASCENDC_CPU_DEBUG
    tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
#else

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *qHashHost, *kHashHost, *topKHost;
    uint8_t *qHashDevice, *kHashDevice, *topKDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
#endif

    // GenerateTilingData(tiling, BLOCK_DIM);
    // uint32_t dataTypeSize = DATA_TYPE_SIZE[reinterpret_cast<HammingTilingData *>(tiling)->dataType];
    // uint32_t xLen = reinterpret_cast<HammingTilingData *>(tiling)->xLen;
    // uint32_t yLen = reinterpret_cast<HammingTilingData *>(tiling)->yLen;
    // uint32_t totalLength = (xLen > yLen)? xLen : yLen;
    // size_t qHashFileSize = xLen * dataTypeSize;
    // size_t kHashFileSize = yLen * dataTypeSize;
    // size_t indexFileSize = totalLength * dataTypeSize;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(qHashFileSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(kHashFileSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(indexFileSize);

    ReadFile("./input/input_x.bin", qHashFileSize, x, qHashFileSize);
    ReadFile("./input/input_y.bin", kHashFileSize, y, kHashFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(hamming_dist_top_k_custom, BLOCK_DIM, x, y, z,
                *reinterpret_cast<HammingTilingData *>(tiling));

    WriteFile("./output/output_z.bin", z, indexFileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclrtMallocHost((void **)(&qHashHost), qHashFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&kHashHost), kHashFileSize));
    CHECK_ACL(aclrtMallocHost((void **)(&topKHost), indexFileSize));
    CHECK_ACL(aclrtMalloc((void **)&qHashDevice, qHashFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&kHashDevice, kHashFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&topKDevice, indexFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_qhash.bin", qHashFileSize, qHashHost, qHashFileSize);
    ReadFile("./input/input_khash.bin", kHashFileSize, kHashHost, kHashFileSize);

    CHECK_ACL(aclrtMemcpy(qHashDevice, qHashFileSize, qHashHost, qHashFileSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(kHashDevice, kHashFileSize, kHashHost, kHashFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(hamming_dist_top_k_custom)(BLOCK_DIM, stream, qHashDevice, kHashDevice, topKDevice,
        reinterpret_cast<HammingTilingData *>(tiling));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // std::cout << "Op time: " << average_time_event << " us" << std::endl;
    //     std::cout << "Peak Flops (No GM bound): " << PEAK_CVEC_FLOPS << " TFlops" << std::endl;
    //     std::cout << "Peak Flops (GM bound): " << peak_gm_bound << " TFlops" << std::endl;
    //     std::cout << "Perf: " << tflops << " / " << peak_gm_bound << " TFlops" << std::endl;
    //     std::cout << "Peak Util (No GM bound): " << std::setprecision(4) << util << "%" << std::endl;
    //     std::cout << "Peak Util (GM bound): " << std::setprecision(4) << util_gm_bound << "%" << std::endl;

    CHECK_ACL(aclrtMemcpy(topKHost, indexFileSize, topKDevice, indexFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_topk_idx.bin", topKHost, indexFileSize);

    CHECK_ACL(aclrtFree(qHashDevice));
    CHECK_ACL(aclrtFree(kHashDevice));
    CHECK_ACL(aclrtFree(topKDevice));
    CHECK_ACL(aclrtFreeHost(qHashHost));
    CHECK_ACL(aclrtFreeHost(kHashHost));
    CHECK_ACL(aclrtFreeHost(topKHost));
    CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
