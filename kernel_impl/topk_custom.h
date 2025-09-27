/**
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the
 * "License"). Please refer to the License for details. You may not use this
 * file except in compliance with the License. THIS SOFTWARE IS PROVIDED ON AN
 * "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS
 * FOR A PARTICULAR PURPOSE. See LICENSE in the root of the software repository
 * for the full text of the License.
 */

#ifndef EXAMPLES_SORT_TOPK_CUSTOM_H
#define EXAMPLES_SORT_TOPK_CUSTOM_H
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace MyCustomKernel {
// struct VecTiling {

//     uint32_t k;
//     uint32_t outter;
//     uint32_t inner;
//     uint32_t n;
//     uint32_t minsize;
//     bool isLargest;
//     TopkTiling topKTilingData;
// };
struct VecTiling {
    // 来自 TopkCustom
    uint32_t k;
    uint32_t outter;
    uint32_t inner;
    uint32_t n;
    uint32_t minsize;
    bool     isLargest;

    // *********** basic info ***********
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t seqLenPad;
    uint32_t seqBlock;

    uint32_t topK;
    uint32_t topKCompressed;
    uint32_t topKComprssedPad; // 注意原拼写

    uint32_t hidDim;
    uint32_t hidDimCompressNum;
    uint32_t hidDimCompressPadNum;
    uint32_t hidDimCompressAddNum;

    uint32_t totalNum;
    uint32_t groupNum;
    uint32_t bufferNum;

    uint32_t scalarSize;

    // *********** Core Offset -- per core -- for GM ***********
    uint32_t qHashCoreOffset;
    uint32_t kHashCoreOffset;
    uint32_t indexCoreOffset;
    uint32_t qHashCoreOffsetBlock;
    uint32_t kHashCoreOffsetBlock;
    uint32_t indexCoreOffsetBlock;

    // *********** tiling info -- elements ***********
    uint32_t seqLenTilingLen;
    uint32_t seqLenTilingNum;
    uint32_t seqLenTilingTailLen;
    uint32_t seqLenBlockNum;

    // Size info -- elements, not byte
    uint32_t qHashTilingSize;
    uint32_t qHashSingleTilingSize;
    uint32_t kHashTilingSize;
    uint32_t kHashSingleTilingSize;

    uint32_t tmpWorkSpaceSize;

    // XOR / rightshift / reduceSum workspace
    uint32_t reduceSumWorkSpaceSize;

    uint32_t hammingXORTilingSize;
    uint32_t hammingXORSingleTilingSize;
    uint32_t hammingRightTilingSize;
    uint32_t hammingRightSingleTilingSize;
    uint32_t hammingCastTilingSize;
    uint32_t hammingCastSingleTilingSize;
    uint32_t hammingLastRowTilingSize;
    uint32_t hammingLastRowSingleTilingSize;
    uint32_t hammingSumTilingSize;
    uint32_t hammingSumSingleTilingSize;
    uint32_t hammingCumTilingSize;
    uint32_t hammingCumSingleTilingSize;
    uint32_t hammingReduceTilingSize;
    uint32_t hammingReduceSingleTilingSize;
    uint32_t hammingResultTilingSize;
    uint32_t hammingResultSingleTilingSize;

    uint32_t resultSize;
    uint32_t resultSingleSize;

    uint32_t resultChunkSize;
    uint32_t resultChunkSingleSize;

    // *********** topK ***********
    uint32_t chunkSize;
    uint32_t chunkRepeat;
    uint32_t chunkTailMask;
    uint32_t chunkMode;
    uint32_t chunkTopKNum;

    uint32_t indexChunkSize;
    uint32_t indexChunkSingleSize;
    uint32_t topKChunkSize;
    uint32_t topKChunkSingleSize;

    // 嵌套的结构体
    TopkTiling topKTilingData;
};


constexpr uint8_t K_FLOAT = 8;
constexpr uint8_t K_HALF = 16;
constexpr uint8_t LOCAL_BYTES = 32;

template <typename hashDataType, typename computeDataType, 
        typename indexDataType, bool isInitIndex = false, 
        bool isHasfinish = true, bool isReuseSrc = false,
        bool topkMode = false, bool tmpLocal = true>
class KernelTopK {
public:
    __aicore__ inline KernelTopK() {}
    __aicore__ inline KernelTopK(const VecTiling& tilingData) : param_(tilingData) {}
    

    /* @brief: 搬入数据
    * Dst tensor, Src tensor
    */
    template <typename T = int16_t>
    __aicore__ inline void DataCopyInCustom(const LocalTensor<T>& dst, 
                                            const GlobalTensor<T>& src, 
                                            int64_t blockLen, int64_t blockCount,
                                            int64_t rightPadding = 0, int64_t paddingValue = 0,
                                            int64_t dstStride = 0, int64_t srcStride = 0){
        
        
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = blockCount;
        dataCopyExtParams.blockLen = blockLen;
        dataCopyExtParams.dstStride = dstStride;
        dataCopyExtParams.srcStride = srcStride;
        
        // 此处每次都需要padding，效率极低
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.rightPadding = 0; // 0 for hashDataType
        dataCopyPadExtParams.paddingValue = T(0); // 0 for hashDataType
        DataCopyPad(dst, src, dataCopyExtParams, dataCopyPadExtParams);
    }

    /* @brief: 搬出数据
    * Dst tensor, Src tensor
    */
    template <typename T = half>
    __aicore__ inline void DataCopyOutCustom(const GlobalTensor<T>& dst, 
                                             const LocalTensor<T>& src, 
                                             int64_t blockLen, 
                                             int64_t blockCount){
        
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = blockCount;
        dataCopyExtParams.blockLen = blockLen;

        DataCopyPad(dst, src, dataCopyExtParams);
    }

    /* @brief: 计算kHash和qHash的距离，通过求XOR和右移看奇偶获取汉明距离
    * output: hammingSumUB
    * input: qHash, kHash,
    *        XOR, rightShift
    *        hammingReduce, hammingSum
    *        scalar
    *        qHash [Group, HDim], SeqLen, curTile
    * 这里有一个在SeqLen上的内循环，以支持
    */
    __aicore__ inline void Hamming(uint32_t group, uint32_t HDim, uint32_t seqLen, uint32_t curTile){

        // test
        
        // // Deque from outside
        AscendC::LocalTensor<hashDataType> qHash = qHashUB.DeQue<hashDataType>();
        AscendC::LocalTensor<hashDataType> kHash = kHashUB.DeQue<hashDataType>();
        AscendC::LocalTensor<hashDataType> scalar = scalarUB.DeQue<hashDataType>();

        AscendC::LocalTensor<computeDataType> result = resultUB.DeQue<computeDataType>();
        
        AscendC::LocalTensor<hashDataType> XORRightTmp = XORRightTmpUB.AllocTensor<hashDataType>();
        auto XOR = XORRightTmp;
        auto rightShift = XORRightTmp[param_.hammingXORSingleTilingSize];
        auto tmp = XORRightTmp[2 * param_.hammingXORSingleTilingSize];
        // AscendC::LocalTensor<hashDataType> XOR = XORUB.AllocTensor<hashDataType>();
        // AscendC::LocalTensor<hashDataType> rightShift = hammingRightUB.AllocTensor<hashDataType>();
        // AscendC::LocalTensor<hashDataType> tmp = tmpWorkSpaceUB.AllocTensor<hashDataType>();

        AscendC::LocalTensor<computeDataType> hammingCastCum = hammingCastCumUB.AllocTensor<computeDataType>();
        auto hammingCast = hammingCastCum;
        auto hammingCum = hammingCastCum[param_.hammingCastSingleTilingSize];
        // AscendC::LocalTensor<computeDataType> hammingCast = hammingCastUB.AllocTensor<computeDataType>();
        // AscendC::LocalTensor<computeDataType> hammingCum = hammingCumUB.AllocTensor<computeDataType>();
        AscendC::LocalTensor<computeDataType> hammingLastRow = hammingLastRowUB.AllocTensor<computeDataType>();
        AscendC::LocalTensor<computeDataType> hammingSum = hammingSumUB.AllocTensor<computeDataType>();
        AscendC::LocalTensor<computeDataType> hammingReduce = hammingReduceUB.AllocTensor<computeDataType>();
        // AscendC::LocalTensor<computeDataType> hammingResult = hammingResultUB.AllocTensor<computeDataType>();
        AscendC::LocalTensor<computeDataType> reduceSumWorkSpace = reduceSumWorkSpaceUB.AllocTensor<computeDataType>();

        static constexpr AscendC::CumSumConfig cumSumConfig{false, false, true};
        const AscendC::CumSumInfo cumSumInfo{group, 16};

        // if(GetBlockIdx() == 0){
        //     AscendC::PRINTF("qHash\n");
        //     AscendC::DumpTensor(qHash, 1, 128);
        //     AscendC::PRINTF("kHash\n");
        //     AscendC::DumpTensor(kHash, 1, 128);
        // }

        // TBD 由于后续需要做转置，因此seqlen需要输入进hamming中，并且转置后做一次掩码
        // 每次针对一个seqlen进行操作
        for (uint32_t i = 0; i < seqLen; i++){
            
            // AscendC::PRINTF("%d\n", i);

            for (size_t j = 0; j < group; j++)
            {
                DataCopy(tmp[j * param_.hidDimCompressPadNum], kHash[i * param_.hidDimCompressPadNum], param_.hidDimCompressPadNum);
            }
            PipeBarrier<PIPE_V>();
            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("copy khash\n");
                AscendC::DumpTensor(tmp, 1, 128);
            }
            
            Xor(XOR, qHash, tmp, param_.hidDimCompressPadNum * group);
            PipeBarrier<PIPE_V>();
            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("XOR\n");
                AscendC::DumpTensor(XOR, 1, 128);
            }

            // Hamming compute -- 没有同步问题
            // x = x - ((x >> 1) & 0x5555555555555555ULL);              // 每2位计数
            ShiftRight(rightShift, XOR, (hashDataType)1, group * 16); // rightShift = x >> 1, 只有group个参与,*16是因为有16个元素组成一个datablock
            PipeBarrier<PIPE_V>();
            And(rightShift, rightShift, scalar[0], group * 16); // scalar[0-8] = 0x5555555555555555ULL 
            PipeBarrier<PIPE_V>();
            Sub(XOR, XOR, rightShift, group * 16); // XOR = x - ((x >> 1) & 0x5555555555555555ULL)
            PipeBarrier<PIPE_V>();
            
            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("2 bit\n");
                AscendC::DumpTensor(XOR, 1, 128);
            }

            // x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL); // 每4位计数
            ShiftRight(rightShift, XOR, (hashDataType)2, group * 16);
            PipeBarrier<PIPE_V>();
            And(rightShift, rightShift, scalar[16 * group], group * 16); // scalar[16-24] = 0x3333333333333333ULL
            PipeBarrier<PIPE_V>();
            And(XOR, XOR, scalar[16 * group], group * 16); // scalar[16] = 0x3333333333333333ULL
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16); // XOR = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL)
            PipeBarrier<PIPE_V>();

            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("4 bit\n");
                AscendC::DumpTensor(XOR, 1, 128);
            }

            // x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;               // 每8位计数
            ShiftRight(rightShift, XOR, (hashDataType)4, group * 16);
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();
            And(XOR, XOR, scalar[32 * group], group * 16); // scalar[32-40] = 0x0F0F0F0F0F0F0F0ULL
            PipeBarrier<PIPE_V>();

            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("8 bit\n");
                AscendC::DumpTensor(XOR, 1, 128);
            }

            // x = x + (x >> 8);                                        // 每16位
            ShiftRight(rightShift, XOR, (hashDataType)8, group * 16);
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();

            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("16 bit\n");
                AscendC::DumpTensor(XOR, 1, 128);
            }

            // x = x & 0x1F;                             // 最终结果
            And(XOR, XOR, scalar[48 * group], group * 16);       // scalar[48-56] = 0x000000000000007F
            PipeBarrier<PIPE_V>();

            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("final\n");
                AscendC::DumpTensor(XOR, 1, 128);
            }

            // 计算完一个SeqLen的Hamming，接下来进行Cast -- sync error
            AscendC::RoundMode roundMode = AscendC::RoundMode::CAST_ROUND;
            Cast(hammingCast, XOR, roundMode, group * 16); // hammingLastRow [1, 16] -- 16是DATABLOCKLEN
            PipeBarrier<PIPE_V>();

            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("cast\n");
                AscendC::DumpTensor(hammingCast, 1, 128);
            }

            // 计算完一个SeqLen的Hamming，接下来进行CumSum
            CumSum<computeDataType, cumSumConfig>(hammingCum, hammingLastRow, hammingCast, cumSumInfo);      // hammingSum [T_S, 16] -- 16是DATABLOCKLEN
            PipeBarrier<PIPE_V>();
            
            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("cumsum lastrow\n");
                AscendC::DumpTensor(hammingLastRow, 1, 128);
            }

            // copy hammingLastRow to hammingSum
            Copy(hammingSum[i * 16], hammingLastRow, 8, 1, {0, 0, 0, 0}); //
            PipeBarrier<PIPE_V>();

            if(GetBlockIdx() == 0 && i == 0){
                AscendC::PRINTF("copy sum\n");
                AscendC::DumpTensor(hammingSum, 1, 128);
            }
        }

        // if(GetBlockIdx() == 0){
        //     // [SeqLen, DATABLOCKLEN]
        //     AscendC::PRINTF("copy sum total\n");
        //     AscendC::DumpTensor(hammingSum, 1, 128);
        // }

        // 算完cumsum后，需要对sum进行求reducesum [T_seqLen, DATABLOCKLEN] -> [T_seqLen, 1] -- TBD -- 当前假定seqLen都是整数倍
        BlockReduceSum<computeDataType, true>(hammingReduce, hammingSum, (seqLen * 16 + 128 - 1) / 128, 128, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        // // 尾块
        // BlockReduceSum<computeDataType, true>(hammingReduce, hammingSum, seqlen / 8, 128, 8, 8, 8);
        // PipeBarrier<PIPE_V>();

        CopyRepeatParams copyRepeatParams;
        copyRepeatParams.srcStride = 1;
        copyRepeatParams.dstStride = 1;
        copyRepeatParams.srcRepeatSize = 8;
        copyRepeatParams.dstRepeatSize = 8;
        // copy hammingReduce to hammingResult
        Copy(result[curTile * param_.seqLenTilingLen], hammingReduce, 128, (seqLen + 128 - 1) / 128, copyRepeatParams); // hammingResult [G, T_S]
        PipeBarrier<PIPE_V>();

        // // result enque
        resultUB.EnQue<computeDataType>(result);

        // // FreeTensor
        kHashUB.FreeTensor(kHash);
        XORRightTmpUB.FreeTensor(XORRightTmp);
        hammingCastCumUB.FreeTensor(hammingCastCum);
        hammingLastRowUB.FreeTensor(hammingLastRow);
        hammingSumUB.FreeTensor(hammingSum);
        hammingReduceUB.FreeTensor(hammingReduce);
        reduceSumWorkSpaceUB.FreeTensor(reduceSumWorkSpace);

        // // 重复Enque同一个张量，因为Qhash scalar不会改变
        qHashUB.EnQue<hashDataType>(qHash);
        scalarUB.EnQue<hashDataType>(scalar);

    }

    // template <typename T = uint16_t>
    __aicore__ inline void ReduceMaxCustom(const LocalTensor<computeDataType> &outTensor, const LocalTensor<computeDataType> &inTensor, 
                                            const uint8_t chunkSize){

        int32_t totalRepeat = param_.chunkRepeat; /* 8: BlockReduceMax一次并行计算8个dataBlock */
        int32_t repeat = MAX_REPEAT_TIMES < totalRepeat ? MAX_REPEAT_TIMES : totalRepeat;
        int32_t loopNum = matmul::CeilDiv(totalRepeat, repeat);
        int32_t tailRepeat = totalRepeat - (loopNum - 1) * repeat;
        // int32_t chunkRepeatTail = param_.chunkRepeatTail;

        uint64_t mask[2]; /* 2: 逐bit设置mask，需要2个64bit */
        if (chunkSize == 16) { /* chunkSize 只支持16*/
            mask[0] = UINT64_MAX;
            mask[1] = UINT64_MAX;

            uint32_t srcOffset = 0;
            uint32_t dstOffset = 0;
            for (int32_t i = 0; i < loopNum - 1; i++) {
                BlockReduceMax<computeDataType>(outTensor[dstOffset], inTensor[srcOffset], repeat, mask, 1, 1, 8); // 8: srcRepStride
                srcOffset += repeat * 8 * 16; /* 8: BlockReduceMax一次并行计算8个dataBlock, 16: 每个dataBlock有32Bytes，包含16个half的值*/
                dstOffset += repeat * 8; /* 8: BlockReduceMax一次并行计算8个dataBlock, 输出8个点 */
            }
            BlockReduceMax<computeDataType>(outTensor[dstOffset], inTensor[srcOffset], tailRepeat - 1, mask, 1, 1, 8); // 8: srcRepStride
            srcOffset += (tailRepeat - 1) * 8 * 16; 
            dstOffset += (tailRepeat - 1) * 8; 
            BlockReduceMax<computeDataType>(outTensor[dstOffset], inTensor[srcOffset], 1, param_.chunkTailMask, 1, 1, 8); // 8: srcRepStride
        }
        // BlockReduceSum<computeDataType, true>(hammingReduce, hammingSum, (seqLen + 8 - 1) / 8, 128, 8, 8, 8);
    }

    // 此处可通过duplicate优化
    template <typename T>
    __aicore__ inline void InitScalar(){

        AscendC::LocalTensor<T> tensor = scalarUB.AllocTensor<T>();
        
        for (uint32_t i = 0; i < param_.groupNum; i ++){
            AscendC::Duplicate<T>(tensor[i * 16], T(0x5555), 8); // 每个group的前8个datablock设置为0x5555
            AscendC::Duplicate<T>(tensor[i * 16 + 16 * param_.groupNum], T(0x3333), 8); // 每个group的第16-24个datablock设置为0x3333
            AscendC::Duplicate<T>(tensor[i * 16 + 32 * param_.groupNum], T(0x0F0F), 8); // 每个group的第32-40个datablock设置为0x0F0F
            AscendC::Duplicate<T>(tensor[i * 16 + 48 * param_.groupNum], T(0x001F), 8); // 每个group的第48-56个datablock设置为0x001F
        }
        // AscendC::DumpTensor(tensor, 1, 256);
        scalarUB.EnQue<T>(tensor);
    }

    template <typename T>
    __aicore__ inline void CopyQHash(const GlobalTensor<T>& src, 
                                    int64_t blockLen, int64_t blockCount,
                                    int64_t rightPadding = 0, int64_t paddingValue = 0,
                                    int64_t dstStride = 0, int64_t srcStride = 0){

        AscendC::LocalTensor<T> qHashLocal = qHashUB.AllocTensor<T>();

        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = blockCount;
        dataCopyExtParams.blockLen = blockLen;
        dataCopyExtParams.dstStride = dstStride;
        dataCopyExtParams.srcStride = srcStride;

        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.rightPadding = rightPadding; // 0 for hashDataType
        dataCopyPadExtParams.paddingValue = paddingValue; // 0 for hashDataType
        DataCopyPad(qHashLocal, src, dataCopyExtParams, dataCopyPadExtParams); // 这里仍然提供的是待填充数据的起始值

        qHashUB.EnQue<T>(qHashLocal);
    }

    template <typename T>
    __aicore__ inline void CopyKHash(const GlobalTensor<T>& src, 
                                    int64_t blockLen, int64_t blockCount,
                                    int64_t rightPadding = 0, int64_t paddingValue = 0,
                                    int64_t dstStride = 0, int64_t srcStride = 0){

        // AscendC::PRINTF("%d\n", blockLen);
        // AscendC::PRINTF("%d\n", blockCount);
        
        AscendC::LocalTensor<T> kHashLocal = kHashUB.AllocTensor<T>();

        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = blockCount;
        dataCopyExtParams.blockLen = blockLen;
        dataCopyExtParams.dstStride = dstStride;
        dataCopyExtParams.srcStride = srcStride;

        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.rightPadding = rightPadding; // 0 for hashDataType
        dataCopyPadExtParams.paddingValue = paddingValue; // 0 for hashDataType
        DataCopyPad(kHashLocal, src, dataCopyExtParams, dataCopyPadExtParams); // 这里仍然提供的是待填充数据的起始值

        kHashUB.EnQue<T>(kHashLocal);
    }

    /* @brief: 在这里对输入的序列进行Chunk缩减
    * output: outTensor
    * input: inTensor; ChunkSize, chunkNum, chunkTail, ChunkMode
    */
    // template <typename T = uint16_t>
    __aicore__ inline void ChunkCompress(uint32_t chunkSize,
                                         uint32_t chunkMode){
        
        AscendC::LocalTensor<computeDataType> inTensor = resultUB.DeQue<computeDataType>();
        AscendC::LocalTensor<computeDataType> outTensor = resultChunkUB.DeQue<computeDataType>();
        if (chunkMode == 0) { // BlockMax
            ReduceMaxCustom(outTensor, inTensor, static_cast<uint8_t>(chunkSize));
        }
        resultChunkUB.EnQue<computeDataType>(outTensor);
        resultUB.FreeTensor(inTensor);
    }

    __aicore__ inline void CopyTiling(VecTiling tilingData) {

        outter = tilingData.outter;
        inner = tilingData.inner;
        n = tilingData.n;
        isSmallMode = topkMode;
        tmplocalBytes = tilingData.minsize;
        topKTilingData = tilingData.topKTilingData;
        k = tilingData.k;
        // 计算k_pad
        if (sizeof(hashDataType) == sizeof(float)) {
            k_pad = (k + K_FLOAT - 1) / K_FLOAT * K_FLOAT;
        } else {
            k_pad = (k + K_HALF - 1) / K_HALF * K_HALF;
        }
        kpad_index = (k + K_FLOAT) / K_FLOAT * K_FLOAT;
        isLargest = tilingData.isLargest;
        inDataSize = inner * outter;
        outValueDataSize = k_pad * outter;
        outIndexDataSize = kpad_index * outter;

        inputdexDataSize = inner;
        if (topkMode == true) {
            inputdexDataSize = inDataSize;
        }

        finishLocalBytes = outter * sizeof(bool);
        if (finishLocalBytes % LOCAL_BYTES != 0) {
            finishLocalBytes = (finishLocalBytes + LOCAL_BYTES - 1) / LOCAL_BYTES * LOCAL_BYTES;
        }

    }

    __aicore__ inline void Init(GM_ADDR qHash, GM_ADDR kHash,
                            GM_ADDR topKIndex, VecTiling tilingData)
    {

        CopyTiling(tilingData);

        qHashGm.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType*>(qHash));
        kHashGm.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType*>(kHash));
        indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ indexDataType*>(topKIndex));

        AscendC::PRINTF("param_.indexChunkSingleSize: %d\n", param_.indexChunkSingleSize);
        
        // VECIN
        pipe.InitBuffer(qHashUB, 2, sizeof(hashDataType) * tilingData.qHashSingleTilingSize);
        pipe.InitBuffer(kHashUB, 2, sizeof(hashDataType) * tilingData.kHashSingleTilingSize);
        pipe.InitBuffer(scalarUB, 1, sizeof(hashDataType) * tilingData.scalarSize);
        // VECCALC
        pipe.InitBuffer(XORRightTmpUB, 2, 3 * sizeof(hashDataType) * tilingData.hammingXORSingleTilingSize);
        pipe.InitBuffer(hammingCastCumUB, 2, 2 * sizeof(hashDataType) * tilingData.hammingCumSingleTilingSize);
        pipe.InitBuffer(hammingSumUB, 2, sizeof(hashDataType) * tilingData.hammingSumSingleTilingSize);
        pipe.InitBuffer(hammingReduceUB, 2, sizeof(computeDataType) * tilingData.hammingReduceSingleTilingSize);
        pipe.InitBuffer(hammingLastRowUB, 2, sizeof(computeDataType) * tilingData.hammingLastRowSingleTilingSize);
        pipe.InitBuffer(resultUB, 2, sizeof(computeDataType) * tilingData.resultSingleSize);
        pipe.InitBuffer(resultChunkUB, 2, sizeof(computeDataType) * tilingData.resultChunkSingleSize);
        pipe.InitBuffer(reduceSumWorkSpaceUB, 1, sizeof(computeDataType) * tilingData.reduceSumWorkSpaceSize);
        // VECOUT
        pipe.InitBuffer(indexChunkUB, 1, sizeof(indexDataType) * tilingData.indexChunkSingleSize);
        pipe.InitBuffer(topKChunkUB, 2, sizeof(indexDataType) * tilingData.topKChunkSingleSize);


        // srcGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType *>(srcGmValue), inDataSize);
        // srcGlobal2.SetGlobalBuffer(reinterpret_cast<__gm__ indexDataType *>(srcGmIndex), inputdexDataSize);
        // srcGlobal3.SetGlobalBuffer(reinterpret_cast<__gm__ bool *>(finishGm), finishLocalBytes / sizeof(bool));
        // dstGlobal1.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType *>(dstGmValue), outValueDataSize);
        // dstGlobal2.SetGlobalBuffer(reinterpret_cast<__gm__ indexDataType *>(dstGmIndex), outIndexDataSize);

        // pipe.InitBuffer(inQueueX1, 1, inDataSize * sizeof(hashDataType));
        // pipe.InitBuffer(inQueueX2, 1, inputdexDataSize * sizeof(indexDataType));
        // pipe.InitBuffer(inQueueX3, 1, finishLocalBytes);
        // pipe.InitBuffer(outQueueY1, 1, outValueDataSize * sizeof(hashDataType));
        // pipe.InitBuffer(outQueueY2, 1, outIndexDataSize * sizeof(indexDataType));
    }

    __aicore__ inline void Process(){

        uint8_t blocknum = GetBlockNum();
        // AscendC::PRINTF("blocknum: %d\n", blocknum);
        // AscendC::PRINTF("param_.totalNum: %d\n", param_.totalNum);

        // Alloc and EnQue Scalar Tensor;
        InitScalar<hashDataType>();
        AscendC::LocalTensor<computeDataType> result = resultUB.AllocTensor<computeDataType>();
        resultUB.EnQue<computeDataType>(result);
        
        // Alloc and EnQue Index
        AscendC::LocalTensor<indexDataType> indexChunk = indexChunkUB.AllocTensor<indexDataType>();
        ArithProgression<indexDataType>(indexChunk, static_cast<indexDataType>(0), static_cast<indexDataType>(1), static_cast<indexDataType>(param_.indexChunkSize));
        PipeBarrier<PIPE_V>();
        
        AscendC::LocalTensor<computeDataType> resultChunk;
        AscendC::LocalTensor<bool> finishLocal;
        AscendC::LocalTensor<indexDataType> topKChunk;
        // AscendC::LocalTensor<uint8_t> tmpTopKLocal;

        // optiling::TopkTiling

        // 分核
        for (uint32_t core_idx = GetBlockIdx(); core_idx < param_.totalNum; core_idx += blocknum)
        {
            
            int64_t qHashGMOffset = core_idx * param_.qHashCoreOffset;
            int64_t kHashGMOffset = core_idx * param_.kHashCoreOffset;
            int64_t indexGMOffset = core_idx * param_.indexCoreOffset;

            resultChunk = resultChunkUB.AllocTensor<computeDataType>();
            topKChunk = topKChunkUB.AllocTensor<indexDataType>();
            // topKValueChunk = topKValueChunkUB.AllocTensor<computeDataType>();
            resultChunkUB.EnQue<computeDataType>(resultChunk);
            topKChunkUB.EnQue<indexDataType>(topKChunk);
            // topKValueChunkUB.EnQue<computeDataType>(topKValueChunk);

            // 每个迭代仅需要搬运一次QHash即可
            // EnQue QHash
            CopyQHash(qHashGm[qHashGMOffset], 
                int64_t(param_.hidDimCompressNum * sizeof(hashDataType)), // Byte
                int64_t(param_.groupNum), 
                int64_t(param_.hidDimCompressAddNum), 
                0, 0, 0);

            AscendC::PRINTF("%d\n", param_.seqLenTilingNum);

            for (uint32_t curTile = 0; curTile < param_.seqLenTilingNum; curTile++){
                
                // curseqlen可能不是16的倍数
                auto curSeqLen = curTile == param_.seqLenTilingNum - 1 ? param_.seqLenTilingTailLen : param_.seqLenTilingLen;

                auto curKHashOffset = kHashGMOffset + curTile * param_.seqLenTilingLen * param_.hidDimCompressNum;

                // Alloc And EnQue KHash Tensor;
                CopyKHash(kHashGm[curKHashOffset], 
                    int64_t(param_.hidDimCompressNum * sizeof(hashDataType)), // Byte
                    int64_t(curSeqLen), 
                    int64_t(param_.hidDimCompressAddNum), 
                    0, 0, 0);

                // AscendC::LocalTensor<hashDataType> kHashLocal = kHashUB.DeQue<hashDataType>();
                // kHashUB.EnQue<hashDataType>(kHashLocal); // 这里需要重新enque，保证在hamming中可以被使用
                
                // // Hammimng: deque KHash, enque Result
                Hamming(param_.groupNum, param_.hidDimCompressPadNum, curSeqLen, curTile);
                PipeBarrier<PIPE_ALL>();

                // if (core_idx == 0){
                //     AscendC::DumpTensor(kHashLocal, 1, 128);
                // }

                // // Free KHash Tensor -- 最后一次deque，清空kHashUB的队列
                // AscendC::LocalTensor<hashDataType> kHashLocal = kHashUB.DeQue<hashDataType>();
                // kHashUB.FreeTensor(kHashLocal);
            }

            ChunkCompress(param_.chunkSize, param_.chunkMode);
            PipeBarrier<PIPE_ALL>();

            if (core_idx == 0){
                AscendC::PRINTF("resultChunk\n");
                AscendC::DumpTensor(resultChunk, 1, param_.indexChunkSize);
            }
            
            // // DeQue topKChunk, indexChunk, resultChunk
            // // EnQue indexChunk
            // TopKCustom(param_.topKCompressed);
            // // PipeBarrier<PIPE_ALL>();

            // CopyOutIndex(topKChunkUB);
            // PipeBarrier<PIPE_ALL>();

            // Free QHash Tensor -- 最后一次deque，清空qHashUB的队列
            AscendC::LocalTensor<hashDataType> qHashLocal = qHashUB.DeQue<hashDataType>();
            qHashUB.FreeTensor(qHashLocal);
            // resultChunkUB.FreeTensor(resultChunk);
            // topKChunkUB.FreeTensor(topKChunk);

            // PipeBarrier<PIPE_ALL>();
        }

        // Free Tensor;
        // AscendC::LocalTensor<computeDataType> result = resultUB.DeQue<computeDataType>();
        resultUB.FreeTensor(result);
        AscendC::LocalTensor<hashDataType> scalarLocal = scalarUB.DeQue<hashDataType>();
        scalarUB.FreeTensor(scalarLocal);

        // *****************  Old  *****************

    }

    // __aicore__ inline void Process() {
    //     CopyIn();
    //     Compute();
    //     CopyOut();
    // }

private:
    __aicore__ inline void CopyIn() {
        // AscendC::LocalTensor<hashDataType> srcLocalValue = inQueueX1.AllocTensor<hashDataType>();
        // AscendC::LocalTensor<indexDataType> srcLocalIndex = inQueueX2.AllocTensor<indexDataType>();
        // AscendC::LocalTensor<bool> srcLocalFinish = inQueueX3.AllocTensor<bool>();
        // AscendC::DataCopy(srcLocalValue, srcGlobal1, inDataSize);
        // AscendC::DataCopy(srcLocalIndex, srcGlobal2, inputdexDataSize);
        // AscendC::DataCopy(srcLocalFinish, srcGlobal3, finishLocalBytes / sizeof(bool));

        // inQueueX1.EnQue(srcLocalValue);
        // inQueueX2.EnQue(srcLocalIndex);
        // inQueueX3.EnQue(srcLocalFinish);
    }
    __aicore__ inline void Compute() {
        // AscendC::LocalTensor<hashDataType> dstLocalValue = outQueueY1.AllocTensor<hashDataType>();
        // AscendC::LocalTensor<indexDataType> dstLocalIndex = outQueueY2.AllocTensor<indexDataType>();

        // AscendC::LocalTensor<hashDataType> srcLocalValue = inQueueX1.DeQue<hashDataType>();
        // AscendC::LocalTensor<indexDataType> srcLocalIndex = inQueueX2.DeQue<indexDataType>();
        // AscendC::LocalTensor<bool> srcLocalFinish = inQueueX3.DeQue<bool>();

        // auto newTopkMode = AscendC::TopKMode::TOPK_NORMAL;
        // if (isSmallMode) {
        //     newTopkMode = AscendC::TopKMode::TOPK_NSMALL;
        // }
        // auto topKInfo = AscendC::TopKInfo();
        // topKInfo.outter = outter;
        // topKInfo.inner = inner;
        // topKInfo.n = n;

        // hashDataType scalar1(0);
        // AscendC::Duplicate<hashDataType>(dstLocalValue, scalar1, outValueDataSize);
        // indexDataType scalar2(0);
        // AscendC::Duplicate<indexDataType>(dstLocalIndex, scalar2, outIndexDataSize);
        // if (!tmpLocal) {
        //     if (isSmallMode) {
        //         AscendC::TopK<hashDataType, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NSMALL>(dstLocalValue,
        //             dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, k, topKTilingData, topKInfo,
        //             isLargest);
        //     } else {
        //         AscendC::TopK<hashDataType, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NORMAL>(dstLocalValue,
        //             dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, k, topKTilingData, topKInfo,
        //             isLargest);
        //     }
        // } else {
        //     if (tmplocalBytes % LOCAL_BYTES != 0) {
        //         tmplocalBytes = (tmplocalBytes + LOCAL_BYTES - 1) / LOCAL_BYTES * LOCAL_BYTES;
        //     }
        //     pipe.InitBuffer(tmplocalBuf, tmplocalBytes);
        //     AscendC::LocalTensor<uint8_t> tmplocalTensor = tmplocalBuf.Get<uint8_t>();
        //     if (isSmallMode) {
        //         AscendC::TopK<hashDataType, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NSMALL>(dstLocalValue,
        //             dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, tmplocalTensor, 
        //             k, topKTilingData, topKInfo, isLargest);
        //     } else {
        //         AscendC::TopK<hashDataType, isInitIndex, isHasfinish, isReuseSrc, AscendC::TopKMode::TOPK_NORMAL>(dstLocalValue,
        //             dstLocalIndex, srcLocalValue, srcLocalIndex, srcLocalFinish, tmplocalTensor, 
        //             k, topKTilingData, topKInfo, isLargest);
        //     }
        // }

        // outQueueY1.EnQue<hashDataType>(dstLocalValue);
        // outQueueY2.EnQue<indexDataType>(dstLocalIndex);

        // inQueueX1.FreeTensor(srcLocalValue);
        // inQueueX2.FreeTensor(srcLocalIndex);
        // inQueueX3.FreeTensor(srcLocalFinish);
    }
    __aicore__ inline void CopyOut() {
        // AscendC::LocalTensor<hashDataType> dstLocalValue = outQueueY1.DeQue<hashDataType>();
        // AscendC::LocalTensor<indexDataType> dstLocalIndex = outQueueY2.DeQue<indexDataType>();

        // AscendC::DataCopy(dstGlobal1, dstLocalValue, outValueDataSize);
        // AscendC::DataCopy(dstGlobal2, dstLocalIndex, outIndexDataSize);
        // outQueueY1.FreeTensor(dstLocalValue);
        // outQueueY2.FreeTensor(dstLocalIndex);
    }

private:
    // AscendC::GlobalTensor<hashDataType> srcGlobal1;
    // AscendC::GlobalTensor<indexDataType> srcGlobal2;
    // AscendC::GlobalTensor<bool> srcGlobal3;
    // AscendC::GlobalTensor<hashDataType> dstGlobal1;
    // AscendC::GlobalTensor<indexDataType> dstGlobal2;

    AscendC::TPipe pipe;

    // AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX1;
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX2;
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX3;

    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY1;
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY2;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmplocalBuf;

    AscendC::GlobalTensor<hashDataType> qHashGm;
    AscendC::GlobalTensor<hashDataType> kHashGm;
    AscendC::GlobalTensor<indexDataType> indexGm;

    // VECIN
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qHashUB;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> kHashUB;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> scalarUB; // scalar for hamming dist
    AscendC::TQue<AscendC::TPosition::VECIN, 1> XORRightTmpUB; // for hamming dist
    // VECCALC
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingCastCumUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingReduceUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingLastRowUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingSumUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> resultChunkUB; // chunk result
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> reduceSumWorkSpaceUB; // reduce sum workspace
    // VECOUT
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> resultUB; // hamming dist result
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> indexChunkUB; // index for topk
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> topKChunkUB; // topk result


    uint32_t tmplocalBytes = 0;
    uint32_t inDataSize = 0;
    uint32_t inputdexDataSize = 0;
    uint32_t inputdexBytes = 0;
    uint32_t finishLocalBytes;
    uint32_t outValueDataSize = 0;
    uint32_t outIndexDataSize = 0;
    uint32_t k;
    uint32_t k_pad;
    uint32_t kpad_index;
    bool isLargest = true;
    TopkTiling topKTilingData;
    uint32_t outter;
    uint32_t inner;
    uint32_t n;
    bool isSmallMode = false;

    VecTiling param_;
};

} // namespace MyCustomKernel

#endif // EXAMPLES_SORT_TOPK_CUSTOM_H
