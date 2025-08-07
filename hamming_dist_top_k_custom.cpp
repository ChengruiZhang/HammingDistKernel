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

using namespace AscendC

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
    template<int32_t BUFFER_NUM>
    __aicore__ inline void Init(GM_ADDR qHash, GM_ADDR kHash, GM_ADDR index, HammingTilingData tiling)
    {
        // 初始化scalar



        auto blockidx = GetBlockIdx();
        qHashGm.SetGlobalBuffer((__gm__ *hashDataType) qHash);
        kHashGm.SetGlobalBuffer((__gm__ *hashDataType) kHash);
        indexGm.SetGlobalBuffer((__gm__ *int32_t) index);

        // parse tiling factor
        param_.totalNum = tiling.totalNum;
        param_.groupNum = tiling.groupNum;
        param_.chunkSize = tiling.chunkSize;
        param_.chunkNum = tiling.chunkNum;
        param_.compressTopK = tiling.compressTopK;

        // Core Offset
        param_.qHashCoreOffset = tiling.qHashCoreOffset;
        param_.kHashCoreOffset = tiling.kHashCoreOffset;
        param_.indexCoreOffset = tiling.indexCoreOffset;

        // tiling info
        param_.seqLenTilingLen = tiling.seqLenTilingLen;
        param_.seqLenTilingNum = tiling.seqLenTilingNum;    // contain tail
        param_.seqLenTilingTailLen = tiling.seqLenTilingTailLen;
        
        param_.hDimTilingLen = tiling.hDimTilingLen;
        param_.hDimTilingNum = tiling.hDimTilingNum;        // contain tail
        param_.hDimTilingTailLen = tiling.hDimTilingTailLen;

        // Size info -- elements, not byte
        param_.bufferNum = tiling.bufferNum;
        param_.qHashTilingSize = tiling.qHashTilingSize;    // contain buffer num  
        param_.qHashSingleTilingSize = tiling.qHashSingleTilingSize;
        param_.kHashTilingSize = tiling.kHashTilingSize;    // contain buffer num  
        param_.kHashSingleTilingSize = tiling.kHashSingleTilingSize;
        // 注意，这里的qHashGroup为整个hidden Dim的Size，不是分块之后的
        param_.qHashGroupSize = tiling.qHashGroupSize;      // contain buffer num  
        param_.qHashGroupSingleSize = tiling.qHashGroupSingleSize;
        param_.indexChunkTilingSize = tiling.indexChunkTilingSize;                // contain buffer num  
        param_.indexChunkSingleTilingSize = tiling.indexChunkSingleTilingSize;

    }

    /* @brief: 搬入数据
    * Dst tensor, Src tensor
    */
    template <typename T = uint16_t>
    __aicore__ inline DataCopyInCustom(LocalTensor<T>& dst, GlobalTensor<T>& src, 
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
    __aicore__ inline DataCopyOutCustom(GlobalTensor<T>& dst, LocalTensor<T>& src, 
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

    // /* @brief: 处理对qhash的累加
    // * input: qHashRowtensor, qHash tensor, qHash [Dim1Len, Dim2Len], axis, curTile, 
    // */
    // template <typename T = uint16_t>
    // __aicore__ inline void QHashGroup(LocalTensor<T>& outputRowtensor, LocalTensor<T>& qHash, 
    //                                   uint32_t Dim1Len, uint32_t Dim2Len, bool axis, uint32_t curTile){
    //     for (uint32_t i = 0; i < param_.hDimTilingNum; i++){
    //         auto tileLength = i == param_.hDimTilingNum - 1 ? param_.hDimTilingTailLen : param_.hDimTilingLen;   
    //         auto groupLenth = param_.groupNum;
    //         int64_t dstOffset;
    //         int64_t srcOffset;
    //         int64_t blockLen;
    //         int64_t blockNum;
    //         // load qHash
    //         DataCopyCustom(qHash, qHashGm, dstOffset, srcOffset, blockLen, blockNum);
    //         CumSumConfig config;
    //         config.isLastAxis = asix;
    //         config.isReuseSource = true;
    //         config.outputLastRow = true;
    //         CumSumInfo cumSumInfo;
    //         cumSumInfo.outter = Dim1Len;
    //         cumSumInfo.inner = Dim2Len;
    //         CumSum<config>(qHash, outputRowtensor, qHash, cumSumInfo);
    //     }
    // }

    template <typename T = uint16_t>
    __aicore__ inline void XOR(LocalTensor<T>& dst, 
                               LocalTensor<T>& qHash, LocalTensor<T>& kHash, 
                               uint32_t group, uint32_t seqLen){
        for (uint32_t i = 0; i < group; i++){
            Xor(dst[i * param_.hidDimCompressPadNum], qHash[i * param_.hidDimCompressPadNum], kHash[seqLen * param_.hidDimCompressPadNum], param_.hidDimCompressPadNum);
            PipeBarrier(PIPE_V);
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
    __aicore__ inline void Hamming(LocalTensor<T>& hammingResult, 
                                   LocalTensor<T>& qHash, LocalTensor<T>& kHash, 
                                   LocalTensor<T>& XOR, LocalTensor<T>& rightShift,
                                   LocalTensor<T>& hammingReduce, LocalTensor<T>& hammingSum,
                                   LocalTensor<T>& scalar, LocalTensor<T>& reduceSumWorkSpace,
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
            XOR(XOR, qHash, kHash, group, i); // 获得 T_HDimPad * G
            PipeBarrier(PIPE_V);

            // x = x - ((x >> 1) & 0x5555555555555555ULL);              // 每2位计数
            ShiftRight(rightShift, XOR, (T)1, group * 16); // rightShift = x >> 1, 只有group个参与,*16是因为有16个元素组成一个datablock
            PipeBarrier(PIPE_V);
            And(rightShift, rightShift, scalar[0], group * 16); // scalar[0] = 0x5555555555555555ULL
            PipeBarrier(PIPE_V);
            Sub(XOR, XOR, rightShift, group * 16); // XOR = x - ((x >> 1) & 0x5555555555555555ULL)
            PipeBarrier(PIPE_V);
            
            // x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL); // 每4位计数
            ShiftRight(rightShift, XOR, (T)2, group * 16);
            PipeBarrier(PIPE_V);
            And(rightShift, rightShift, scalar[16], group * 16); // scalar[16] = 0x3333333333333333ULL
            PipeBarrier(PIPE_V);
            And(XOR, XOR, scalar[16], group * 16); // scalar[16] = 0x3333333333333333ULL
            PipeBarrier(PIPE_V);
            Add(XOR, XOR, rightShift, group * 16); // XOR = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL)
            PipeBarrier(PIPE_V);

            // x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;               // 每8位计数
            ShiftRight(rightShift, XOR, (T)4, group * 16);
            PipeBarrier(PIPE_V);
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier(PIPE_V);
            And(XOR, XOR, scalar[32], group * 16); // scalar[32] = 0x0F0F0F0F0F0F0F0ULL
            PipeBarrier(PIPE_V);

            // x = x + (x >> 8);                                        // 每16位
            ShiftRight(rightShift, XOR, (T)8, group * 16);
            PipeBarrier(PIPE_V);
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier(PIPE_V);

            // x = x + (x >> 16);                                       // 每32位
            ShiftRight(rightShift, XOR, (T)16, group * 16);
            PipeBarrier(PIPE_V);
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier(PIPE_V);

            // x = x + (x >> 32);                                       // 每64位
            ShiftRight(rightShift, XOR, (T)32, group * 16); // 32是因为每个datablock有16个元素，每个元素是64位
            PipeBarrier(PIPE_V);
            Add(XOR, XOR, rightShift, group * 16);
            PipeBarrier(PIPE_V);

            // x = x & 0x7F;                             // 最终结果
            And(XOR, XOR, scalar[48], group * 16);       // scalar[48] = 0x000000000000007F
            PipeBarrier(PIPE_V);

            // 计算完一个SeqLen的Hamming，接下来进行CumSum
            CumSum<cumSumConfig>(XOR, hammingSum[i * 16], XOR, cumSumInfo);      // hammingSum [T_S, 16] -- 16是DATABLOCKLEN
            PipeBarrier(PIPE_V);

        }

        // 算完cumsum后，需要对sum进行求reducesum
        // ReduceSum(hammingSum, hammingSum, )
        ReduceSum(hammingSum, hammingSum, reduceSumWorkSpace, 16, seqLen, 0);
        PipeBarrier(PIPE_V);

        // Transpose Sum结果，并将首行搬运至 hammingResult
        for (uint32_t i = 0; i < param_.seqLenBlockNum; i++){
            Transpose(hammingSum[i * 256], hammingSum[i * 256]); hammingResult [G, T_S]
        }

        CopyRepeatParams copyRepeatParams;
        copyRepeatParams.srcStride = 0;
        copyRepeatParams.dstStride = 0;
        copyRepeatParams.srcRepeatSize = 16; 
        copyRepeatParams.dstRepeatSize = 0;

        Copy(hammingResult[curTile * param_.seqLenTilingLenPad], hammingSum, 16, seqLenBlockNum - 1, copyRepeatParams); // 将首行搬运至 hammingResult [G, T_S]， 每次搬运的个数为16
        Copy(hammingResult[], hammingSum[], param_.seqLenTilingTailLen, 1, copyRepeatParams); // 尾块 -- TBD
        // Transpose(hammingResult, hammingSum), 
        PipeBarrier(PIPE_V);

    }

    /* @brief: 在这里对输入的序列进行Chunk缩减
    * input: inTensor; ChunkSize, chunkNum, chunkTail, ChunkMode
    * output: outTensor
    */
    template <typename T = uint16_t>
    __aicore__ inline void ChunkCompress(LocalTensor<T>& outTensor, LocalTensor<T>& inTensor,
                                         uint32_t chunkSize, uint32_t chunkNum,  
                                         uint32_t chunkTail, uint32_t chunkMode){
        
    }

    /* @brief: 在这里计算topk
    * output: dstIndexLocal
    * input: srcValueLocal, srcIndexLocal, k, 
    */
    template <typename T>
    __aicore__ inline void TopKCustom(const LocalTensor<int32_t> &dstIndexLocal,
                                      const LocalTensor<T> &srcValueLocal, const LocalTensor<int32_t> &srcIndexLocal,
                                      const int32_t k, 
        // 下面两个参数仍需修改
        const HammingDistTopKTilingData &tiling, uint32_t n)
    {
        LocalTensor<bool> finishLocal;
        TopKInfo topkInfo;
        topkInfo.outter = tiling.params.outter;
        topkInfo.n = n;
        topkInfo.inner = matmul::CeilDiv(n, 32) * 32; /* 32: inner has to be aligned to 32 */
        TopK<half, true, false, false, TopKMode::TOPK_NORMAL>(dstValueLocal, dstIndexLocal, srcValueLocal, srcIndexLocal, finishLocal, k, tiling.topkTiling, topkInfo, true);
    }

    template <typename T = half>
    __aicore__ inline void SetTensorAddr(LocalTensor<T>& tensor, uint32_t dataLen, uint32_t bufferAddr, uint8_t logicPos){
        TBuffAddr TBuffAddr_;
        TBuffAddr_.dataLen = dataLen;
        TBuffAddr_.bufferAddr = bufferAddr;
        TBuffAddr_.logicPos = logicPos;
        tensor.SetAddr(TBuffAddr_);
    }

    // 此处不确定能否对tensor cast
    template <typename T = uint16_t>
    __aicore__ inline void SetScalarValue(LocalTensor<T>& tensor){
        // 0x5555555555555555
        for (uint32_t i = 0; i < 4; i++){ tensor.SetValue(i, uint16_t(0x5555)); }
        // 0x3333333333333333
        for (uint32_t i = 8; i < 12; i++){tensor.SetValue(i, uint16_t(0x3333));}
        // 0x0F0F
        for (uint32_t i = 16; i < 20; i++){tensor.SetValue(i, uint16_t(0x0F0F));}
        // 0x000000000000007F
        tensor.SetValue(15, uint16_t(0x007F));
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
        // int64_t hammingReduceBaseUB = rightShiftBaseUB + param_.hammingRightTilingSize;
        // int64_t hammingSumBaseUB = hammingReduceBaseUB + param_.hammingReduceTilingSize;
        int64_t hammingResultBaseUB = hammingReduceBaseUB + param_.hammingReduceTilingSize;
        int64_t hammingChunkBaseUB = hammingResultBaseUB + param_.hammingResultTilingSize;
        int64_t reduceSumWorkSpaceBaseUB = hammingChunkBaseUB + param_.hammingChunkTilingSize;
        // int64_t hammingReduceBaseUB = hammingReduceBaseUB + param_.hammingReduceTilingSize;
        // VECOUT
        int64_t indexChunkBaseUB = 0;
        int64_t topKChunkBaseUB = indexChunkBaseUB + param_.indexChunkTilingSize;

        LocalTensor<int16_t> qHashUB, kHashUB, // input
                             scalarUB, reduceSumWorkSpaceUB, // scalar
                             XORUB, rightShiftUB, // hamming intermediate
                             hammingReduceUB, hammingSumUB, hammingResultUB, hammingChunkUB, // hamming result 
                             indexChunkUB, topKChunkUB; // result
        
        SetScalarValue(scalarUB);

        // 此处设置的时候就需要考虑 multi buffer
        // VECIN
        SetTensorAddr<hashDataType>(qHashUB, param_.qHashTilingSize, qHashBaseUB, 9);
        SetTensorAddr<hashDataType>(kHashUB, param_.kHashTilingSize, kHashBaseUB, 9);
        // VECIN -- Scalar
        SetTensorAddr<hashDataType>(scalarUB, param_.scalarSize, scalarBaseUB, 9);
        // VECALC
        SetTensorAddr<hashDataType>(XORUB, param_.hammingGroupTilingSize, XORBaseUB, 11);
        SetTensorAddr<hashDataType>(rightShiftUB, param_.hammingGroupTilingSize, rightShiftBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingSumUB, param_.hammingSumTilingSize, hammingSumBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingReduceUB, param_.hammingGroupTilingSize, hammingReduceBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingResultUB, param_.hammingResultTilingSize, hammingResultBaseUB, 11);
        SetTensorAddr<hashDataType>(hammingChunkUB, param_.hammingChunkTilingSize, hammingChunkBaseUB, 11); 
        SetTensorAddr<hashDataType>(reduceSumWorkSpaceUB, param_.reduceSumWorkSpace, reduceSumWorkSpaceBaseUB, 11); 
        
        // VECOUT
        SetTensorAddr<hashDataType>(indexChunkUB, param_.indexChunkTilingSize, indexChunkBaseUB, 10);
        SetTensorAddr<hashDataType>(topKChunkUB, param_.topKChunkTilingSize, topKChunkBaseUB, 10);
        
        GenerateIndex(indexChunkUB, 0, param_.chunkSize, param_.chunkNum); // ArithProgression or CreateVecIndex
        PipeBarrier(PIPE_V);
        GenerateScalar(scalarUB); // scalar for hamming dist
        PipeBarrier(PIPE_V);

        for (uint32_t core_idx = GetBlockIdx(); core_idx < totalNum; core_idx += blocknum){
            
            int64_t qHashGMOffset = core_idx * qHashCoreOffset;
            int64_t kHashGMOffset = core_idx * kHashCoreOffset;
            int64_t indexGMOffset = core_idx * indexCoreOffset;
            
            // auto kHashOffset = loop_ping_flag ? kHashBaseUB + kHashSingleTilingSize : kHashBaseUB;
            // VECIN
            auto qHashUBOffset = loop_ping_flag ? qHashBaseUB + param_.qHashSingleTilingSize : qHashBaseUB;
            auto kHashUBOffset = loop_ping_flag ? kHashBaseUB + param_.kHashSingleTilingSize : kHashBaseUB;
            // VECALC
            auto XORUBOffset = loop_ping_flag ? XORBaseUB + param_.hammingXORSingleTilingSize : XORBaseUB;
            auto rightShiftUB = loop_ping_flag ? rightShiftBaseUB + param_.hammingRightSingleTilingSize : rightShiftBaseUB;
            auto hammingSumUBOffset = loop_ping_flag ? hammingSumBaseUB + param_.hammingSumSingleTilingSize : hammingSumBaseUB;
            auto hammingReduceUBOffset = loop_ping_flag ? hammingReduceBaseUB + param_.hammingReduceSingleTilingSize : hammingReduceBaseUB;
            auto hammingResultUBOffset = loop_ping_flag ? hammingResultBaseUB + param_.hammingResultSingleTilingSize : hammingResultBaseUB;
            auto hammingChunkUBOffset = loop_ping_flag ? hammingChunkBaseUB + param_.hammingChunkSingleTilingSize : hammingChunkBaseUB;
            // VECOUT
            auto indexChunkUBOffset = loop_ping_flag ? indexChunkBaseUB + param_.indexChunkSingleTilingSize : indexChunkBaseUB;
            auto topKChunkUBOffset = loop_ping_flag ? topKChunkUB + param_.topKChunkSingleTilingSize : topKChunkUB;

            // copyin qHash -- groupNum, HDim
            DataCopyInCustom(qHash[qHashUBOffset], qHashGm[qHashGMOffset], 
                             param_.hidDimCompressNum * sizeof(hashDataType), param_.groupNum, 
                             param_.hidDimCompressAddNum * sizeof(hashDataType), 0,
                             0, 0);

            // T_SeqLen 维度循环
            for (uint32_t seqLenTiling = 0; seqLenTiling < param_.seqLenTilingNum; seqLenTiling += 1){
                
                auto curSeqLen = seqLenTiling == seqLenTilingNum - 1 ? param_.seqLenTilingTailLen : param_.seqLenTilingLen;
                
                // copyin kHash -- T_SeqLen, HDim
                DataCopyInCustom(kHash[kHashUBOffset], kHashGM[kHashGMOffset], 
                                 param_.hidDimCompressNum * sizeof(hashDataType), curSeqLen, 
                                 param_.hidDimCompressAddNum * sizeof(hashDataType), 0,
                                 0, 0);
                
                // 先计算Hamming dist -- 实际hDimTilingNum = 1
                Hamming(hammingResultUB[hammingResultUBOffset + seqLenTiling * param_.hammingResultSingleTilingSize], 
                        qHashUB[qHashUBOffset], kHashUB[kHashUBOffset], 
                        XORUB[XORUBOffset], hammingRightUB[hammingRightUBOffset], 
                        hammingReduceUB[hammingReduceUBOffset], hammingSumUB[hammingSumUBOffset],
                        scalarUB, reduceSumWorkSpaceUB, 
                        param_.groupNum, param_.hidDimCompressPadNum, curSeqLen, seqLenTiling);
                PipeBarrier(PIPE_V);

            }

            // 此处有尾块问题，需要mask -- 引入tail
            ChunkCompress(hammingChunkUB[hammingChunkUBOffset], 
                          hammingResultUB[hammingResultUBOffset + seqLenTiling * hammingResultSingleTilingSize],
                          param_.chunkSize, param_.chunkNum, 
                          param_.chunkTail, param_.chunkMode); // [1, chunkNum]
            PipeBarrier(PIPE_V);

            TopKCustom(topKChunkUB[topKChunkUBOffset], hammingChunkUB[hammingChunkUBOffset], indexChunkUB[indexChunkUBOffset]); // [1, compressTopK]
            PipeBarrier(PIPE_V);

            // get output GM offset
            DataCopyOutCustom(indexGm[indexCoreOffset], indexChunkUB[indexChunkUBOffset],
                           indexCoreOffsetBlock, 1);

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
    int32_t bufferNum = tiling.bufferNum;
    assert(bufferNum <= 2);

    KernelHammingDistTopK<uint16_t> op;
    op.Init<bufferNum>(qHash, kHash, index, tiling);
    op.Process();

}
