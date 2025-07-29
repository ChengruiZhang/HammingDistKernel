/**
 * @file Hamming_custom_tiling.h
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
    
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t totalNum; // totalNum = batchSize * seqLen

    uint32_t groupNum; // headQ / headK

    uint32_t formerCoreBlockLength;
    uint32_t formerCoreBlockNum;

    uint32_t tailCoreBlockNum;
    uint32_t tailCoreBlockLength;

    uint32_t xLen;
    uint32_t yLen;
    uint32_t coef;
    uint32_t axis;
    uint32_t dataType;

    uint32_t isEvenCore;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;

    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formerTileNum;
    uint32_t formerTileLength;
    uint32_t formerLastTileLength;

    uint32_t tailNum; 
    uint32_t tailLength;
    uint32_t tailTileNum;
    uint32_t tailTileLength;
    uint32_t tailLastTileLength;
};
#endif