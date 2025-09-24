/**
 * @file hamming_dist_top_k_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "hamming_dist_top_k_custom_tiling.h"
#include "hamming_dist_top_k_custom_base.h"
#include "kernel_operator.h"

constexpr int32_t UB_SIZE = 256; // 256 KB

using namespace AscendC;

template <typename hashDataType, typename computeDataType, typename indexDataType> 
class KernelHammingDistTopK {
public:
    __aicore__ inline KernelHammingDistTopK() {}

    /* @brief:
    * 输入qHash: [B, Hk, G, HDim] ND格式存放，kHash: [B, Hk, SeqLen, HDim] ND格式存放
    * 在B和Hk维度分核，每一个核对于qHash的地址变化为 qHashCoreOffset=G*HDim*sizeof(type)
    * 对于kHash的变化为 kHashCoreOffset=SeqLen*HDim*sizeof(type)
    * 
    */
    __aicore__ inline void Init(GM_ADDR qHash, GM_ADDR kHash, GM_ADDR index, HammingTilingData tiling)
    {
        // 初始化scalar
        // auto blockidx = GetBlockIdx();
        qHashGm.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType*>(qHash));
        kHashGm.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType*>(kHash));
        indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ indexDataType*>(index));

        // VECIN
        pipe.InitBuffer(qHashUB, 2, sizeof(hashDataType) * tiling.qHashSingleTilingSize);
        pipe.InitBuffer(kHashUB, 2, sizeof(hashDataType) * tiling.kHashSingleTilingSize);
        pipe.InitBuffer(scalarUB, 1, sizeof(hashDataType) * tiling.scalarSize);
        // VECCALC
        pipe.InitBuffer(XORRightTmpUB, 2, 3 * sizeof(hashDataType) * tiling.hammingXORSingleTilingSize);
        // pipe.InitBuffer(XORUB, 2, sizeof(hashDataType) * tiling.hammingXORSingleTilingSize);
        // pipe.InitBuffer(hammingRightUB, 2, sizeof(hashDataType) * tiling.hammingRightSingleTilingSize);
        // pipe.InitBuffer(tmpWorkSpaceUB, 2, sizeof(hashDataType) * tiling.tmpWorkSpaceSize); // G * hidDimCompressPadNum
        pipe.InitBuffer(hammingCastCumUB, 2, 2 * sizeof(hashDataType) * tiling.hammingCumSingleTilingSize);
        // pipe.InitBuffer(hammingCastUB, 2, sizeof(computeDataType) * tiling.hammingCastSingleTilingSize);
        // pipe.InitBuffer(hammingCumUB, 2, sizeof(computeDataType) * tiling.hammingCumSingleTilingSize);
        pipe.InitBuffer(hammingSumUB, 2, sizeof(hashDataType) * tiling.hammingSumSingleTilingSize);
        pipe.InitBuffer(hammingReduceUB, 2, sizeof(computeDataType) * tiling.hammingReduceSingleTilingSize);
        pipe.InitBuffer(hammingLastRowUB, 2, sizeof(computeDataType) * tiling.hammingLastRowSingleTilingSize);
        pipe.InitBuffer(resultUB, 2, sizeof(computeDataType) * tiling.resultSingleSize);
        pipe.InitBuffer(resultChunkUB, 2, sizeof(computeDataType) * tiling.resultChunkSingleSize);
        pipe.InitBuffer(reduceSumWorkSpaceUB, 1, sizeof(computeDataType) * tiling.reduceSumWorkSpaceSize);
        // VECOUT
        pipe.InitBuffer(indexChunkUB, 1, sizeof(indexDataType) * tiling.indexChunkSingleSize);
        pipe.InitBuffer(topKChunkUB, 2, sizeof(indexDataType) * tiling.topKChunkSingleSize);
        // pipe.InitBuffer(topKValueChunkUB, 2, sizeof(computeDataType) * tiling.topKChunkSingleSize);

        param_.batchSize              = tiling.batchSize;
        param_.seqLen                 = tiling.seqLen;
        param_.seqLenPad              = tiling.seqLenPad;
        param_.seqBlock               = tiling.seqBlock;

        param_.topK                   = tiling.topK;
        param_.topKCompressed         = tiling.topKCompressed;
        param_.topKComprssedPad       = tiling.topKComprssedPad;

        param_.hidDim                 = tiling.hidDim;
        param_.hidDimCompressNum      = tiling.hidDimCompressNum;
        param_.hidDimCompressPadNum   = tiling.hidDimCompressPadNum;
        param_.hidDimCompressAddNum   = tiling.hidDimCompressAddNum;

        param_.totalNum               = tiling.totalNum;
        param_.groupNum               = tiling.groupNum;
        param_.bufferNum              = tiling.bufferNum;
        param_.scalarSize             = tiling.scalarSize;

        param_.qHashCoreOffset        = tiling.qHashCoreOffset;
        param_.kHashCoreOffset        = tiling.kHashCoreOffset;
        param_.indexCoreOffset        = tiling.indexCoreOffset;
        param_.qHashCoreOffsetBlock   = tiling.qHashCoreOffsetBlock;
        param_.kHashCoreOffsetBlock   = tiling.kHashCoreOffsetBlock;
        param_.indexCoreOffsetBlock   = tiling.indexCoreOffsetBlock;

        param_.seqLenTilingLen        = tiling.seqLenTilingLen;
        param_.seqLenTilingNum        = tiling.seqLenTilingNum;
        param_.seqLenTilingTailLen    = tiling.seqLenTilingTailLen;
        param_.seqLenBlockNum         = tiling.seqLenBlockNum;

        param_.qHashTilingSize        = tiling.qHashTilingSize;
        param_.qHashSingleTilingSize  = tiling.qHashSingleTilingSize;
        param_.kHashTilingSize        = tiling.kHashTilingSize;
        param_.kHashSingleTilingSize  = tiling.kHashSingleTilingSize;

        param_.tmpWorkSpaceSize       = tiling.tmpWorkSpaceSize;
        param_.reduceSumWorkSpaceSize = tiling.reduceSumWorkSpaceSize;

        param_.hammingXORTilingSize         = tiling.hammingXORTilingSize;
        param_.hammingXORSingleTilingSize   = tiling.hammingXORSingleTilingSize;
        param_.hammingRightTilingSize       = tiling.hammingRightTilingSize;
        param_.hammingRightSingleTilingSize = tiling.hammingRightSingleTilingSize;
        param_.hammingCastTilingSize        = tiling.hammingCastTilingSize;
        param_.hammingCastSingleTilingSize  = tiling.hammingCastSingleTilingSize;
        param_.hammingLastRowTilingSize     = tiling.hammingLastRowTilingSize;
        param_.hammingLastRowSingleTilingSize = tiling.hammingLastRowSingleTilingSize;
        param_.hammingSumTilingSize         = tiling.hammingSumTilingSize;
        param_.hammingSumSingleTilingSize   = tiling.hammingSumSingleTilingSize;
        param_.hammingCumTilingSize         = tiling.hammingCumTilingSize;
        param_.hammingCumSingleTilingSize   = tiling.hammingCumSingleTilingSize;
        param_.hammingReduceTilingSize      = tiling.hammingReduceTilingSize;
        param_.hammingReduceSingleTilingSize= tiling.hammingReduceSingleTilingSize;
        param_.hammingResultTilingSize      = tiling.hammingResultTilingSize;
        param_.hammingResultSingleTilingSize= tiling.hammingResultSingleTilingSize;

        param_.resultSize              = tiling.resultSize;
        param_.resultSingleSize        = tiling.resultSingleSize;
        param_.resultChunkSize         = tiling.resultChunkSize;
        param_.resultChunkSingleSize   = tiling.resultChunkSingleSize;

        param_.chunkSize               = tiling.chunkSize;
        param_.chunkRepeat             = tiling.chunkRepeat;
        param_.chunkTailMask           = tiling.chunkTailMask;
        param_.chunkMode               = tiling.chunkMode;
        param_.chunkTopKNum            = tiling.chunkTopKNum;

        param_.indexChunkSize          = tiling.indexChunkSize;
        param_.indexChunkSingleSize    = tiling.indexChunkSingleSize;
        param_.topKChunkSize           = tiling.topKChunkSize;
        param_.topKChunkSingleSize     = tiling.topKChunkSingleSize;

    }

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

    // template <typename T = int16_t>
    __aicore__ inline void XORCustom(const LocalTensor<hashDataType>& dst, 
                               const LocalTensor<hashDataType>& qHash, const LocalTensor<hashDataType>& kHash, 
                               const LocalTensor<hashDataType>& tmp, uint32_t group, uint32_t seqLen){

        // duplicate kHash to tmp
        for (size_t i = 0; i < group; i++)
        {
            DataCopy(tmp[i * param_.hidDimCompressPadNum], kHash[seqLen * param_.hidDimCompressPadNum], param_.hidDimCompressPadNum);
        }
        PipeBarrier<PIPE_V>();
        
        Xor(dst, qHash, tmp, param_.hidDimCompressPadNum * group);
        PipeBarrier<PIPE_V>();

        // for (uint32_t i = 0; i < group; i++){
        //     Xor(dst[i * param_.hidDimCompressPadNum], qHash[i * param_.hidDimCompressPadNum], kHash[seqLen * param_.hidDimCompressPadNum], param_.hidDimCompressPadNum);
        //     PipeBarrier<PIPE_V>();
        // }
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
            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("copy khash\n");
            //     AscendC::DumpTensor(tmp, 1, 128);
            // }
            
            Xor(XOR, qHash, tmp, param_.hidDimCompressPadNum * group);
            PipeBarrier<PIPE_V>();
            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("XOR\n");
            //     AscendC::DumpTensor(XOR, 1, 128);
            // }

            // Hamming compute -- 没有同步问题
            // x = x - ((x >> 1) & 0x5555555555555555ULL);              // 每2位计数
            ShiftRight(rightShift, XOR, (hashDataType)1, group * 16); // rightShift = x >> 1, 只有group个参与,*16是因为有16个元素组成一个datablock
            PipeBarrier<PIPE_V>();
            And(rightShift, rightShift, scalar[0], group * 16); // scalar[0-8] = 0x5555555555555555ULL 
            PipeBarrier<PIPE_V>();
            Sub(XOR, XOR, rightShift, group * 16); // XOR = x - ((x >> 1) & 0x5555555555555555ULL)
            PipeBarrier<PIPE_V>();
            
            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("2 bit\n");
            //     AscendC::DumpTensor(XOR, 1, 128);
            // }

            // x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL); // 每4位计数
            ShiftRight(rightShift, XOR, (hashDataType)2, group * 16);
            PipeBarrier<PIPE_V>();
            And(rightShift, rightShift, scalar[16 * group], group * 16); // scalar[16-24] = 0x3333333333333333ULL
            PipeBarrier<PIPE_V>();
            And(XOR, XOR, scalar[16 * group], group * 16); // scalar[16] = 0x3333333333333333ULL
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16); // XOR = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL)
            PipeBarrier<PIPE_V>();

            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("4 bit\n");
            //     AscendC::DumpTensor(XOR, 1, 128);
            // }

            // x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;               // 每8位计数
            ShiftRight(rightShift, XOR, (hashDataType)4, group * 16);
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();
            And(XOR, XOR, scalar[32 * group], group * 16); // scalar[32-40] = 0x0F0F0F0F0F0F0F0ULL
            PipeBarrier<PIPE_V>();

            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("8 bit\n");
            //     AscendC::DumpTensor(XOR, 1, 128);
            // }

            // x = x + (x >> 8);                                        // 每16位
            ShiftRight(rightShift, XOR, (hashDataType)8, group * 16);
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();

            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("16 bit\n");
            //     AscendC::DumpTensor(XOR, 1, 128);
            // }

            // x = x & 0x1F;                             // 最终结果
            And(XOR, XOR, scalar[48 * group], group * 16);       // scalar[48-56] = 0x000000000000007F
            PipeBarrier<PIPE_V>();

            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("final\n");
            //     AscendC::DumpTensor(XOR, 1, 128);
            // }

            // 计算完一个SeqLen的Hamming，接下来进行Cast -- sync error
            AscendC::RoundMode roundMode = AscendC::RoundMode::CAST_ROUND;
            Cast(hammingCast, XOR, roundMode, group * 16); // hammingLastRow [1, 16] -- 16是DATABLOCKLEN
            PipeBarrier<PIPE_V>();

            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("cast\n");
            //     AscendC::DumpTensor(hammingCast, 1, 128);
            // }

            // 计算完一个SeqLen的Hamming，接下来进行CumSum
            CumSum<computeDataType, cumSumConfig>(hammingCum, hammingLastRow, hammingCast, cumSumInfo);      // hammingSum [T_S, 16] -- 16是DATABLOCKLEN
            PipeBarrier<PIPE_V>();
            
            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("cumsum lastrow\n");
            //     AscendC::DumpTensor(hammingLastRow, 1, 128);
            // }


            // copy hammingLastRow to hammingSum
            Copy(hammingSum[i * 16], hammingLastRow, 8, 1, {0, 0, 0, 0}); //
            PipeBarrier<PIPE_V>();

            // if(GetBlockIdx() == 0 && i == 0){
            //     AscendC::PRINTF("copy sum\n");
            //     AscendC::DumpTensor(hammingSum, 1, 128);
            // }
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

        // if(GetBlockIdx() == 0){
        //     AscendC::PRINTF("reduce\n");
        //     AscendC::DumpTensor(hammingReduce, 1, 128);
        // }

        CopyRepeatParams copyRepeatParams;
        copyRepeatParams.srcStride = 1;
        copyRepeatParams.dstStride = 1;
        copyRepeatParams.srcRepeatSize = 8;
        copyRepeatParams.dstRepeatSize = 8;
        // copy hammingReduce to hammingResult
        Copy(result[curTile * param_.seqLenTilingLen], hammingReduce, 128, (seqLen + 128 - 1) / 128, copyRepeatParams); // hammingResult [G, T_S]
        PipeBarrier<PIPE_V>();

        // if(GetBlockIdx() == 0){
        //     AscendC::PRINTF("result\n");
        //     AscendC::DumpTensor(result, 1, 128);
        // }

        // // result enque
        resultUB.EnQue<computeDataType>(result);

        // // FreeTensor
        kHashUB.FreeTensor(kHash);
        XORRightTmpUB.FreeTensor(XORRightTmp);
        // XORUB.FreeTensor(XOR);
        // hammingRightUB.FreeTensor(rightShift);
        // tmpWorkSpaceUB.FreeTensor(tmp);
        hammingCastCumUB.FreeTensor(hammingCastCum);
        // hammingCastUB.FreeTensor(hammingCast);
        // hammingCumUB.FreeTensor(hammingCum);
        hammingLastRowUB.FreeTensor(hammingLastRow);
        hammingSumUB.FreeTensor(hammingSum);
        hammingReduceUB.FreeTensor(hammingReduce);
        reduceSumWorkSpaceUB.FreeTensor(reduceSumWorkSpace);

        // // 重复Enque同一个张量，因为Qhash scalar不会改变
        qHashUB.EnQue<hashDataType>(qHash);
        scalarUB.EnQue<hashDataType>(scalar);

        // // Transpose Sum结果，并将首行搬运至 hammingResult
        // for (uint32_t i = 0; i < param_.seqLenBlockNum; i++){
        //     Transpose(hammingSum[i * 256], hammingSum[i * 256]); // hammingResult [G, T_S]
        // }

        // CopyRepeatParams copyRepeatParams;
        // copyRepeatParams.srcStride = 0;
        // copyRepeatParams.dstStride = 0;
        // copyRepeatParams.srcRepeatSize = 16; 
        // copyRepeatParams.dstRepeatSize = 0;

        // Copy(hammingResult[curTile * param_.seqLenTilingLenPad], hammingSum, 16, param_.seqLenBlockNum - 1, copyRepeatParams); // 将首行搬运至 hammingResult [G, T_S]， 每次搬运的个数为16
        // Copy(hammingResult[curTile * param_.seqLenTilingLenPad + (param_.seqLenBlockNum - 1) * param_.seqLenTilingLenPad], hammingSum[(param_.seqLenBlockNum - 1) * 256], param_.seqLenTilingTailLen, 1, copyRepeatParams); // 尾块 -- TBD
        // // Transpose(hammingResult, hammingSum), 
        // PipeBarrier<PIPE_V>();

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


    template <typename T = half>
    __aicore__ inline void SetTensorAddr(LocalTensor<T>& tensor, uint32_t dataLen, uint32_t bufferAddr, uint8_t logicPos){
        TBuffAddr TBuffAddr_;
        TBuffAddr_.dataLen = dataLen;
        TBuffAddr_.bufferAddr = bufferAddr;
        TBuffAddr_.logicPos = logicPos;
        tensor.SetAddr(TBuffAddr_);
    }

    // template <typename T = int16_t>
    __aicore__ inline void GenerateIndex(const LocalTensor<indexDataType>& tensor, int16_t start, int16_t step, uint32_t count){
        ArithProgression<indexDataType>(tensor, static_cast<indexDataType>(start), static_cast<indexDataType>(step), static_cast<int32_t >(count));
        PipeBarrier<PIPE_ALL>();
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

    // __aicore__ inline void TopKCustom(const LocalTensor<computeDataType> &dstValueLocal, 
    //                                 const LocalTensor<indexDataType> &dstIndexLocal, 
    //                                 const LocalTensor<computeDataType> &srcValueLocal, 
    //                                 const LocalTensor<indexDataType> &srcIndexLocal, 
    //                                 const int32_t k, 
    //                                 const HammingDistTopKTilingData &tiling, 
    //                                 uint32_t n)
    // {
    //     LocalTensor<bool> finishLocal;
    //     AscendC::TopKInfo topkInfo;
    //     topkInfo.outter = tiling.params.outter;
    //     topkInfo.n = n;
    //     topkInfo.inner = matmul::CeilDiv(n, 32) * 32; /* 32: inner has to be aligned to 32 */
    //     TopK<half, true, false, false, TopKMode::TOPK_NORMAL>(dstValueLocal, dstIndexLocal, srcValueLocal, srcIndexLocal, finishLocal, k, tiling.topkTiling, topkInfo, true);
    // }

    // __aicore__ inline void TopKCustom(uint16_t topK){

    //     AscendC::LocalTensor<computeDataType> resultChunk = resultChunkUB.DeQue<computeDataType>();
    //     AscendC::LocalTensor<indexDataType> indexChunk = indexChunkUB.DeQue<indexDataType>();
    //     AscendC::LocalTensor<indexDataType> topKChunk = topKChunkUB.DeQue<indexDataType>();

    //     TopK<computeDataType>(topKChunk, resultChunk, topK, axis, false);
    //     PipeBarrier<PIPE_ALL>();

    //     topKChunkUB.EnQue<indexDataType>(topKChunk);
    //     topKValueChunkUB.EnQue<indexDataType>(topKValueChunk);
    //     // resultChunkUB.FreeTensor(resultChunk);

    // }

    /* @brief:
    * 第一版的分块策略为，针对每一个核处理的qHash[G, HDim] kHash[SeqLen, HDim]，在Seqlen维度和HDim维度进行分块，不对qhash的G进行分块
    */
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

            // AscendC::PRINTF("%d\n", param_.seqLenTilingNum);

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

            // if (core_idx == 0){
            //     AscendC::PRINTF("resultChunk\n");
            //     AscendC::DumpTensor(resultChunk, 1, param_.indexChunkSize);
            // }
            
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

private:

    AscendC::TPipe pipe;
    
    // VECIN
    AscendC::TQue<AscendC::TPosition::VECIN, 1> qHashUB;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> kHashUB;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> scalarUB; // scalar for hamming dist
    AscendC::TQue<AscendC::TPosition::VECIN, 1> XORRightTmpUB; // for hamming dist
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> hammingXORUB; // for hamming dist
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> hammingRightUB; // for hamming dist
    // AscendC::TQue<AscendC::TPosition::VECIN, 1> tmpWorkSpaceUB; // tmp workspace
    // VECCALC
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingCastCumUB; // for hamming dist
    // AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingCumUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingReduceUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingLastRowUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingSumUB; // for hamming dist
    // AscendC::TQue<AscendC::TPosition::VECCALC, 1> hammingCastUB; // for hamming dist
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> resultChunkUB; // chunk result
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> reduceSumWorkSpaceUB; // reduce sum workspace
    // VECOUT
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> resultUB; // hamming dist result
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> indexChunkUB; // index for topk
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> topKChunkUB; // topk result
    // AscendC::TQue<AscendC::TPosition::VECOUT, 1> topKValueChunkUB; // topk result
    
    TilingParam param_;

    AscendC::GlobalTensor<hashDataType> qHashGm;
    AscendC::GlobalTensor<hashDataType> kHashGm;
    AscendC::GlobalTensor<indexDataType> indexGm;

    AscendC::GlobalTensor<hashDataType> scalarGm; // scalar for hamming dist
};

extern "C" __global__ __aicore__ void hamming_dist_top_k_custom(GM_ADDR qHash, GM_ADDR kHash, GM_ADDR index, HammingTilingData tiling)
{
    const int32_t bufferNum = tiling.bufferNum;
    assert(bufferNum <= 2);
    // PRINTF("bufferNum: %d\n", bufferNum);

    KernelHammingDistTopK<int16_t, half, int32_t> op;
    op.Init(qHash, kHash, index, tiling);
    op.Process();
}
