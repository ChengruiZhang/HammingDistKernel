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

template <typename hashDataType> 
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
        auto blockidx = GetBlockIdx();
        qHashGm.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType*>(qHash));
        kHashGm.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType*>(kHash));
        indexGm.SetGlobalBuffer(reinterpret_cast<__gm__ hashDataType*>(index));

        // parse tiling factor
        // basic info
        param_.batchSize = tiling.batchSize;
        param_.seqLen = tiling.seqLen;
        param_.seqLenPad = tiling.seqLenPad; // pad elements to 32 Bytes
        param_.seqBlock = (tiling.seqLenPad + 15) / 16; // ceil(seqLenPad / 16) -- 每个datablock有16个元素
        param_.reduceSumWorkSpace = tiling.reduceSumWorkSpace; // reduceSum的工作空间，32 Byte对齐
        param_.hidDim = tiling.hidDim;
        param_.hidDimCompressNum = tiling.hidDimCompressNum; // bool -- int16/int8 压缩之后的实际elements大小
        param_.hidDimCompressPadNum = tiling.hidDimCompressPadNum; // pad to 32 Bytes -- elements大小
        param_.hidDimCompressAddNum = tiling.hidDimCompressAddNum; // pad增加的element个数  
        param_.totalNum = tiling.totalNum; // totalNum = batchSize * headK
        param_.groupNum = tiling.groupNum; // headQ / headK
        param_.chunkSize = tiling.chunkSize; // topk compress block -- 这里会有非整数问题
        param_.chunkRepeat = (param_.seqBlock + 7) / 8; // ceil(seqBlock / 8)
        param_.chunkTailMask = tiling.chunkTailMask; // (seqLen % (8 * 16)) -- 可能为0  
        param_.chunkNum = (tiling.seqLen + param_.chunkSize - 1) / param_.chunkSize; // ceil(SeqLen / chunkSize)  -- 似乎无用
        param_.chunkTail = tiling.seqLen % param_.chunkSize; // SeqLen % chunkSize  -- 似乎无用
        param_.chunkMode = tiling.chunkMode;  // max
        param_.chunkTopKNum = tiling.chunkTopKNum; // need add assert TopK/chunkSize
        param_.scalarSize = tiling.scalarSize; // hamming Scalar -- 4 * 64 = 256 bits = 64 bytes = 32 elements
        // Core Offset -- per core
        param_.qHashCoreOffset = tiling.qHashCoreOffset; // G * hidDimCompressPadNum * sizeof(hashDataType)
        param_.kHashCoreOffset = tiling.kHashCoreOffset; // G * hidDimCompressPadNum * sizeof(hashDataType)
        param_.indexCoreOffset = tiling.indexCoreOffset;
        param_.qHashCoreOffsetBlock = tiling.qHashCoreOffsetBlock; // 32 Byte = 1 block
        param_.kHashCoreOffsetBlock = tiling.kHashCoreOffsetBlock;
        param_.indexCoreOffsetBlock = tiling.indexCoreOffsetBlock;
        // tiling info -- elements
        param_.seqLenTilingLen = tiling.seqLenTilingLen;        
        param_.seqLenTilingLenPad = tiling.seqLenTilingLenPad; // 在尾块中补齐，尾块前为正常
        param_.seqLenTilingNum = tiling.seqLenTilingNum;
        param_.seqLenTilingTailLen = tiling.seqLenTilingTailLen;
        param_.seqLenBlockNum = tiling.seqLenBlockNum; // seqLenTilingLenPad / 16
        // 这里的hDim均为压缩后的
        param_.hDimTilingLen = tiling.hDimTilingLen;
        param_.hDimTilingNum = tiling.hDimTilingNum; // contain tail
        param_.hDimTilingTailLen = tiling.hDimTilingTailLen;
        // Size info -- elements, not byte
        param_.bufferNum = tiling.bufferNum;
        param_.qHashTilingSize = tiling.qHashTilingSize; // contain buffer num -- 一次性读完全部的，HDim切块为1 -- G*bfn*hidDimCompressPadNum
        param_.qHashSingleTilingSize = tiling.qHashSingleTilingSize; // G*hidDimCompressPadNum
        param_.kHashTilingSize = tiling.kHashTilingSize; // contain buffer num -- T_SeqLenPad * hidDimCompressPadNum * bfn
        param_.kHashSingleTilingSize = tiling.kHashSingleTilingSize; // T_SeqLenPad * hidDimCompressPadNum
        
        // hamming -- 2个临时空间足矣
        //     unsigned int popcount16(unsigned int x) {
        //     x = x - ((x >> 1) & 0x5555);               // 每2位
        //     x = (x & 0x3333) + ((x >> 2) & 0x3333);    // 每4位
        //     x = (x + (x >> 4)) & 0x0F0F;               // 每8位
        //     x = (x + (x >> 8)) & 0x001F;               // 合并到16位
        //     return x;
        // }
        // XOR rightshift 这些都是对T_SeqLen中的一个做的
        param_.hammingXORTilingSize = tiling.hammingXORTilingSize; // contain buffer num and tiling -- G * hidDimCompressPadNum * bfn
        param_.hammingXORSingleTilingSize = tiling.hammingXORSingleTilingSize; // G * hidDimCompressPadNum
        param_.hammingRightTilingSize = tiling.hammingRightTilingSize; // G * hidDimCompressPadNum * bfn    
        param_.hammingRightSingleTilingSize = tiling.hammingRightSingleTilingSize; // G * hidDimCompressPadNum
        param_.hammingSumTilingSize = tiling.hammingSumTilingSize; // contain buffer num and tiling -- T_SeqLenPad * DATABLOCKLEN * bfn
        param_.hammingSumSingleTilingSize = tiling.hammingSumSingleTilingSize; // T_SeqLenPad * DATABLOCKLEN
        param_.hammingReduceTilingSize = tiling.hammingReduceTilingSize; // G * 16 (DATABLOCKLEN, 按0扩充至32 Byte) * bfn
        param_.hammingReduceSingleTilingSize = tiling.hammingReduceSingleTilingSize; // G * 16 (DATABLOCKLEN)
        param_.hammingResultTilingSize = tiling.hammingResultTilingSize; // contain buffer num and tiling -- T_SeqLenPad * bfn
        param_.hammingResultSingleTilingSize = tiling.hammingResultSingleTilingSize; // T_SeqLenPad
        param_.hammingChunkTilingSize = tiling.hammingChunkTilingSize; // T_SeqLenPad / 16 * bfn -- block reduce 能够做维度缩减
        param_.hammingChunkSingleTilingSize = tiling.hammingChunkSingleTilingSize; // T_SeqLenPad / 16
        param_.indexChunkTilingSize = tiling.indexChunkTilingSize;  //  ((seqLenPad + 128 - 1) / 128 * 8 + 16 - 1) * 16 * bfn
        param_.indexChunkSingleTilingSize = tiling.indexChunkSingleTilingSize;
        param_.topKChunkTilingSize = tiling.topKChunkTilingSize; // (k + 16 - 1) * 16 * bfn
        param_.topKChunkSingleTilingSize = tiling.topKChunkSingleTilingSize;
    }

    /* @brief: 搬入数据
    * Dst tensor, Src tensor
    */
    template <typename T = uint16_t>
    __aicore__ inline void DataCopyInCustom(const LocalTensor<T>& dst, const GlobalTensor<T>& src, 
                                     int64_t blockLen, int64_t blockCount,
                                     int64_t rightPadding = 0, int64_t paddingValue = 0,
                                     int64_t dstStride = 0, int64_t srcStride = 0){
        
        
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = blockCount;
        dataCopyExtParams.blockLen = blockLen;
        dataCopyExtParams.dstStride = dstStride;
        dataCopyExtParams.srcStride = srcStride;

        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.rightPadding = 0; // 0 for hashDataType
        dataCopyPadExtParams.paddingValue = 0; // 0 for hashDataType
        DataCopyPad(dst, src, dataCopyExtParams, dataCopyPadExtParams);
    }

    /* @brief: 搬出数据
    * Dst tensor, Src tensor
    */
    template <typename T = uint16_t>
    __aicore__ inline void DataCopyOutCustom(const GlobalTensor<T>& dst, const LocalTensor<T>& src, 
                                    //  int64_t dstOffset, int64_t srcOffset, 
                                     int64_t blockLen, int64_t blockCount,
                                     int64_t rightPadding = 0, int64_t paddingValue = 0,
                                     int64_t dstStride = 0, int64_t srcStride = 0){
        
        DataCopyExtParams dataCopyExtParams;
        dataCopyExtParams.blockCount = blockCount;
        dataCopyExtParams.blockLen = blockLen;
        dataCopyExtParams.dstStride = dstStride;
        dataCopyExtParams.srcStride = srcStride;

        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.rightPadding = 0; // 0 for hashDataType
        dataCopyPadExtParams.paddingValue = 0; // 0 for hashDataType
        DataCopyPad(dst, src, dataCopyExtParams, dataCopyPadExtParams);
    }


    /* @brief: 对[M, N]的向量进行除法
    * input: srcTensor, M, N, scalar, axis
    * output: dstTensor
    */
    template <typename T = uint16_t>
    __aicore__ inline void TensorDiv(){
        // 扩充
    }

    template <typename T = uint16_t>
    __aicore__ inline void XORCustom(const LocalTensor<T>& dst, 
                               const LocalTensor<T>& qHash, const LocalTensor<T>& kHash, 
                               uint32_t group, uint32_t seqLen){
        for (uint32_t i = 0; i < group; i++){
            Xor(dst[i * param_.hidDimCompressPadNum], qHash[i * param_.hidDimCompressPadNum], kHash[seqLen * param_.hidDimCompressPadNum], param_.hidDimCompressPadNum);
            PipeBarrier<PIPE_V>();
        }
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
    template <typename T = uint16_t>
    __aicore__ inline void Hamming(const LocalTensor<T>& hammingResult, 
                                   const LocalTensor<T>& qHash, const LocalTensor<T>& kHash, 
                                   const LocalTensor<T>& XOR, const LocalTensor<T>& rightShift,
                                   const LocalTensor<T>& hammingReduce, const LocalTensor<T>& hammingSum,
                                   const LocalTensor<T>& scalar, const LocalTensor<T>& reduceSumWorkSpace,
                                   uint32_t group, uint32_t HDim, uint32_t seqLen, uint32_t curTile){
        

        CumSumConfig cumSumConfig;
        cumSumConfig.isLastAxis = true; // 在最后一个维度上做累加
        cumSumConfig.isReuseSource = true; // 复用source
        cumSumConfig.outputLastRow = true; // 输出最后一行
        
        CumSumInfo cumSumInfo;
        cumSumInfo.outter = group; // G
        cumSumInfo.inner = 16; // DATABLOCKLEN -- 每个datablock有16个元素 

        // TBD 由于后续需要做转置，因此seqlen需要输入进hamming中，并且转置后做一次掩码
        // 每次针对一个seqlen进行操作
        for (uint32_t i = 0; i < seqLen; i++){
            // compute Hamming
            XORCustom(XOR, qHash, kHash, group, i); // 获得 T_HDimPad * G
            PipeBarrier<PIPE_V>();

            // x = x - ((x >> 1) & 0x5555555555555555ULL);              // 每2位计数
            ShiftRight(rightShift, XOR, (T)1, group * 16); // rightShift = x >> 1, 只有group个参与,*16是因为有16个元素组成一个datablock
            PipeBarrier<PIPE_V>();
            And(rightShift, rightShift, scalar[0], group * 16); // scalar[0-8] = 0x5555555555555555ULL 
            PipeBarrier<PIPE_V>();
            Sub(XOR, XOR, rightShift, group * 16); // XOR = x - ((x >> 1) & 0x5555555555555555ULL)
            PipeBarrier<PIPE_V>();
            
            // x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL); // 每4位计数
            ShiftRight(rightShift, XOR, (T)2, group * 16);
            PipeBarrier<PIPE_V>();
            And(rightShift, rightShift, scalar[16], group * 16); // scalar[16-24] = 0x3333333333333333ULL
            PipeBarrier<PIPE_V>();
            And(XOR, XOR, scalar[16], group * 16); // scalar[16] = 0x3333333333333333ULL
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16); // XOR = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL)
            PipeBarrier<PIPE_V>();

            // x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;               // 每8位计数
            ShiftRight(rightShift, XOR, (T)4, group * 16);
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();
            And(XOR, XOR, scalar[32], group * 16); // scalar[32-40] = 0x0F0F0F0F0F0F0F0ULL
            PipeBarrier<PIPE_V>();

            // x = x + (x >> 8);                                        // 每16位
            ShiftRight(rightShift, XOR, (T)8, group * 16);
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();

            // x = x + (x >> 16);                                       // 每32位
            ShiftRight(rightShift, XOR, (T)16, group * 16);
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();

            // x = x + (x >> 32);                                       // 每64位
            ShiftRight(rightShift, XOR, (T)32, group * 16); // 32是因为每个datablock有16个元素，每个元素是64位
            PipeBarrier<PIPE_V>();
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier<PIPE_V>();

            // x = x & 0x7F;                             // 最终结果
            And(XOR, XOR, scalar[48], group * 16);       // scalar[48-56] = 0x000000000000007F
            PipeBarrier<PIPE_V>();

            // 计算完一个SeqLen的Hamming，接下来进行CumSum
            CumSum<hashDataType, cumSumConfig>(XOR, hammingSum[i * 16], XOR, cumSumInfo);      // hammingSum [T_S, 16] -- 16是DATABLOCKLEN
            PipeBarrier<PIPE_V>();

        }

        // 算完cumsum后，需要对sum进行求reducesum
        // ReduceSum(hammingSum, hammingSum, )
        ReduceSum(hammingSum, hammingSum, reduceSumWorkSpace, 16, seqLen, 0);
        PipeBarrier<PIPE_V>();

        // Transpose Sum结果，并将首行搬运至 hammingResult
        for (uint32_t i = 0; i < param_.seqLenBlockNum; i++){
            Transpose(hammingSum[i * 256], hammingSum[i * 256]); // hammingResult [G, T_S]
        }

        CopyRepeatParams copyRepeatParams;
        copyRepeatParams.srcStride = 0;
        copyRepeatParams.dstStride = 0;
        copyRepeatParams.srcRepeatSize = 16; 
        copyRepeatParams.dstRepeatSize = 0;

        Copy(hammingResult[curTile * param_.seqLenTilingLenPad], hammingSum, 16, param_.seqLenBlockNum - 1, copyRepeatParams); // 将首行搬运至 hammingResult [G, T_S]， 每次搬运的个数为16
        Copy(hammingResult[curTile * param_.seqLenTilingLenPad + (param_.seqLenBlockNum - 1) * param_.seqLenTilingLenPad], hammingSum[(param_.seqLenBlockNum - 1) * 256], param_.seqLenTilingTailLen, 1, copyRepeatParams); // 尾块 -- TBD
        // Transpose(hammingResult, hammingSum), 
        PipeBarrier<PIPE_V>();

    }

    template <typename T = uint16_t>
    __aicore__ inline void ReduceMaxCustom(const LocalTensor<T> &outTensor, const LocalTensor<T> &inTensor, 
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
                BlockReduceMax<T>(outTensor[dstOffset], inTensor[srcOffset], repeat, mask, 1, 1, 8); // 8: srcRepStride
                srcOffset += repeat * 8 * 16; /* 8: BlockReduceMax一次并行计算8个dataBlock, 16: 每个dataBlock有32Bytes，包含16个half的值*/
                dstOffset += repeat * 8; /* 8: BlockReduceMax一次并行计算8个dataBlock, 输出8个点 */
            }
            BlockReduceMax<T>(outTensor[dstOffset], inTensor[srcOffset], tailRepeat - 1, mask, 1, 1, 8); // 8: srcRepStride
            srcOffset += (tailRepeat - 1) * 8 * 16; 
            dstOffset += (tailRepeat - 1) * 8; 
            BlockReduceMax<T>(outTensor[dstOffset], inTensor[srcOffset], 1, param_.chunkTailMask, 1, 1, 8); // 8: srcRepStride
        }
        //  else if (chunkSize == 8) { /* chunkSize 只支持1 8 16*/
        //     mask[0] = 0x00ff00ff00ff00ff;
        //     mask[1] = 0x00ff00ff00ff00ff;
        // }

}

    /* @brief: 在这里对输入的序列进行Chunk缩减
    * output: outTensor
    * input: inTensor; ChunkSize, chunkNum, chunkTail, ChunkMode
    */
    template <typename T = uint16_t>
    __aicore__ inline void ChunkCompress(const LocalTensor<T>& outTensor, const LocalTensor<T>& inTensor, uint32_t chunkSize,
                                         uint32_t chunkMode){
        if (chunkMode == 0) { // BlockMax
            ReduceMaxCustom(outTensor, inTensor, static_cast<uint8_t>(chunkSize));
        }
    }

    // /* @brief: 在这里计算topk
    // * output: dstIndexLocal
    // * input: srcValueLocal, srcIndexLocal, k, 
    // */
    // template <typename T>
    // __aicore__ inline void TopKCustom(const LocalTensor<int32_t> &dstIndexLocal,
    //                                   const LocalTensor<T> &srcValueLocal, const LocalTensor<int32_t> &srcIndexLocal,
    //                                   const int32_t k, 
    //     // 下面两个参数仍需修改
    //     const HammingDistTopKTilingData &tiling, uint32_t n)
    // {
    //     LocalTensor<bool> finishLocal;
    //     TopKInfo topkInfo;
    //     topkInfo.outter = tiling.params.outter;
    //     topkInfo.n = n;
    //     topkInfo.inner = matmul::CeilDiv(n, 32) * 32; /* 32: inner has to be aligned to 32 */
    //     TopK<half, true, false, false, TopKMode::TOPK_NORMAL>(dstValueLocal, dstIndexLocal, srcValueLocal, srcIndexLocal, finishLocal, k, tiling.topkTiling, topkInfo, true);
    // }

    template <typename T = half>
    __aicore__ inline void SetTensorAddr(LocalTensor<T>& tensor, uint32_t dataLen, uint32_t bufferAddr, uint8_t logicPos){
        TBuffAddr TBuffAddr_;
        TBuffAddr_.dataLen = dataLen;
        TBuffAddr_.bufferAddr = bufferAddr;
        TBuffAddr_.logicPos = logicPos;
        tensor.SetAddr(TBuffAddr_);
    }


    __aicore__ inline void GenerateIndex(const LocalTensor<T>& tensor, uint32_t start, uint32_t end, uint32_t step){
        // 生成一个从start到end的index
        for (uint32_t i = start; i < end; i += step){
            tensor.SetValue(i / step, i);
        }

    }
    // 此处不确定能否对tensor cast
    template <typename T = uint16_t>
    __aicore__ inline void SetScalarValue(LocalTensor<T>& tensor){
        // 0x5555555555555555
        for (uint32_t i = 0; i < 4; i++){ tensor.SetValue(i, uint16_t(0x5555)); }
        // 0x3333333333333333
        for (uint32_t i = 8; i < 12; i++){tensor.SetValue(i, uint16_t(0x3333));}
        // 0x0F0F0F0F0F0F0F0F
        for (uint32_t i = 16; i < 20; i++){tensor.SetValue(i, uint16_t(0x0F0F));}
        // 0x000000000000007F
        tensor.SetValue(27, uint16_t(0x007F));
    }

    /* @brief:
    * 第一版的分块策略为，针对每一个核处理的qHash[G, HDim] kHash[SeqLen, HDim]，在Seqlen维度和HDim维度进行分块，不对qhash的G进行分块
    * 
    */
    __aicore__ inline void Process()
    {

        uint8_t blocknum = GetBlockNum();
        int32_t loop_ping_flag = param_.bufferNum - 1;

        // VECIN
        int64_t qHashBaseUB = 0;
        int64_t kHashBaseUB = qHashBaseUB + param_.qHashTilingSize;
        int64_t scalarBaseUB = kHashBaseUB + param_.kHashTilingSize;
        // VECALC
        int64_t XORBaseUB = 0;
        int64_t rightShiftBaseUB = XORBaseUB + param_.hammingXORTilingSize;
        int64_t hammingSumBaseUB = rightShiftBaseUB + param_.hammingRightTilingSize;
        int64_t hammingReduceBaseUB = hammingSumBaseUB + param_.hammingSumTilingSize;
        int64_t hammingResultBaseUB = hammingReduceBaseUB + param_.hammingReduceTilingSize;
        int64_t hammingChunkBaseUB = hammingResultBaseUB + param_.hammingResultTilingSize;
        int64_t reduceSumWorkSpaceBaseUB = hammingChunkBaseUB + param_.hammingChunkTilingSize;
        // VECOUT
        int64_t indexChunkBaseUB = 0;
        int64_t topKChunkBaseUB = indexChunkBaseUB + param_.indexChunkTilingSize;

        LocalTensor<uint16_t> qHashUB, kHashUB, // input
                             scalarUB, reduceSumWorkSpaceUB, // scalar
                             XORUB, hammingRightUB, // hamming intermediate
                             hammingReduceUB, hammingSumUB, hammingResultUB, hammingChunkUB, // hamming result 
                             indexChunkUB, topKChunkUB; // result
        

        // 此处设置的时候就需要考虑 multi buffer
        // VECIN
        SetTensorAddr<hashDataType>(qHashUB, param_.qHashTilingSize, qHashBaseUB, 9);
        SetTensorAddr<hashDataType>(kHashUB, param_.kHashTilingSize, kHashBaseUB, 9);
        // VECIN -- Scalar
        SetTensorAddr<hashDataType>(scalarUB, param_.scalarSize, scalarBaseUB, 9);
        // VECALC
        SetTensorAddr<hashDataType>(XORUB, param_.hammingXORTilingSize, XORBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingRightUB, param_.hammingRightTilingSize, rightShiftBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingSumUB, param_.hammingSumTilingSize, hammingSumBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingReduceUB, param_.hammingReduceTilingSize, hammingReduceBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingResultUB, param_.hammingResultTilingSize, hammingResultBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingChunkUB, param_.hammingChunkTilingSize, hammingChunkBaseUB, 11); 
        SetTensorAddr<hashDataType>(reduceSumWorkSpaceUB, param_.reduceSumWorkSpace, reduceSumWorkSpaceBaseUB, 11); 
        // VECOUT
        SetTensorAddr<hashDataType>(indexChunkUB, param_.indexChunkTilingSize, indexChunkBaseUB, 10);
        SetTensorAddr<hashDataType>(topKChunkUB, param_.topKChunkTilingSize, topKChunkBaseUB, 10);
        
        GenerateIndex(indexChunkUB, 0, param_.chunkSize, param_.chunkNum); // ArithProgression or CreateVecIndex -- TBD
        PipeBarrier<PIPE_V>();
        // GenerateScalar(scalarUB); // scalar for hamming dist
        SetScalarValue(scalarUB);
        PipeBarrier<PIPE_V>();

        for (uint32_t core_idx = GetBlockIdx(); core_idx < param_.totalNum; core_idx += blocknum){
            
            int64_t qHashGMOffset = core_idx * param_.qHashCoreOffset;
            int64_t kHashGMOffset = core_idx * param_.kHashCoreOffset;
            int64_t indexGMOffset = core_idx * param_.indexCoreOffset;
            
            // auto kHashOffset = loop_ping_flag ? kHashBaseUB + kHashSingleTilingSize : kHashBaseUB;
            // VECIN
            auto qHashUBOffset = loop_ping_flag ? qHashBaseUB + param_.qHashSingleTilingSize : qHashBaseUB;
            auto kHashUBOffset = loop_ping_flag ? kHashBaseUB + param_.kHashSingleTilingSize : kHashBaseUB;
            // VECALC
            auto XORUBOffset = loop_ping_flag ? XORBaseUB + param_.hammingXORSingleTilingSize : XORBaseUB;
            auto hammingRightUBOffset = loop_ping_flag ? rightShiftBaseUB + param_.hammingRightSingleTilingSize : rightShiftBaseUB;
            auto hammingSumUBOffset = loop_ping_flag ? hammingSumBaseUB + param_.hammingSumSingleTilingSize : hammingSumBaseUB;
            auto hammingReduceUBOffset = loop_ping_flag ? hammingReduceBaseUB + param_.hammingReduceSingleTilingSize : hammingReduceBaseUB;
            auto hammingResultUBOffset = loop_ping_flag ? hammingResultBaseUB + param_.hammingResultSingleTilingSize : hammingResultBaseUB;
            auto hammingChunkUBOffset = loop_ping_flag ? hammingChunkBaseUB + param_.hammingChunkSingleTilingSize : hammingChunkBaseUB;
            // VECOUT
            auto indexChunkUBOffset = loop_ping_flag ? indexChunkBaseUB + param_.indexChunkSingleTilingSize : indexChunkBaseUB;
            auto topKChunkUBOffset = loop_ping_flag ? topKChunkBaseUB + param_.topKChunkSingleTilingSize : topKChunkBaseUB;

            // copyin qHash -- groupNum, HDim
            DataCopyInCustom(qHashUB[qHashUBOffset], qHashGm[qHashGMOffset], 
                             int64_t(param_.hidDimCompressNum * sizeof(hashDataType)), int64_t(param_.groupNum), 
                             int64_t(param_.hidDimCompressAddNum * sizeof(hashDataType)), 0,
                             0, 0);

            // T_SeqLen 维度循环
            for (uint32_t seqLenTiling = 0; seqLenTiling < param_.seqLenTilingNum; seqLenTiling += 1){
                
                auto curSeqLen = seqLenTiling == param_.seqLenTilingNum - 1 ? param_.seqLenTilingTailLen : param_.seqLenTilingLen;
                
                // copyin kHash -- T_SeqLen, HDim
                DataCopyInCustom(kHashUB[kHashUBOffset], kHashGm[kHashGMOffset], 
                                 int64_t(param_.hidDimCompressNum * sizeof(hashDataType)), int64_t(curSeqLen), 
                                 int64_t(param_.hidDimCompressAddNum * sizeof(hashDataType)), 0,
                                 0, 0);
                
                // 先计算Hamming dist -- 实际hDimTilingNum = 1
                Hamming(hammingResultUB[hammingResultUBOffset + seqLenTiling * param_.hammingResultSingleTilingSize], 
                        qHashUB[qHashUBOffset], kHashUB[kHashUBOffset], 
                        XORUB[XORUBOffset], hammingRightUB[hammingRightUBOffset], 
                        hammingReduceUB[hammingReduceUBOffset], hammingSumUB[hammingSumUBOffset],
                        scalarUB, reduceSumWorkSpaceUB, 
                        param_.groupNum, param_.hidDimCompressPadNum, curSeqLen, seqLenTiling);
                PipeBarrier<PIPE_V>();
            }

            // 此处有尾块问题，需要mask -- 引入tail
            ChunkCompress(hammingChunkUB[hammingChunkUBOffset], 
                          hammingResultUB[hammingResultUBOffset],
                          param_.chunkSize,
                          param_.chunkMode); // [1, chunkNum]
            PipeBarrier<PIPE_V>();

            // TopKCustom(topKChunkUB[topKChunkUBOffset], hammingChunkUB[hammingChunkUBOffset], indexChunkUB[indexChunkUBOffset]); // [1, compressTopK]
            // PipeBarrier<PIPE_V>();

            // // get output GM offset
            // DataCopyOutCustom(indexGm[indexCoreOffset], indexChunkUB[indexChunkUBOffset],
            //                indexCoreOffsetBlock, 1); // TBD

            loop_ping_flag = 1 - loop_ping_flag;
        }
    }

private:

    TilingParam param_;

    GlobalTensor<hashDataType> qHashGm;
    GlobalTensor<hashDataType> kHashGm;
    GlobalTensor<hashDataType> indexGm;

    GlobalTensor<hashDataType> scalarGm; // scalar for hamming dist
};

extern "C" __global__ __aicore__ void hamming_dist_top_k_custom(GM_ADDR qHash, GM_ADDR kHash, GM_ADDR index, HammingTilingData tiling)
{
    const int32_t bufferNum = tiling.bufferNum;
    assert(bufferNum <= 2);

    KernelHammingDistTopK<uint16_t> op;
    op.Init(qHash, kHash, index, tiling);
    op.Process();

}
