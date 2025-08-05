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
                                     int64_t dstOffset, int64_t srcOffset, 
                                     int64_t blockLen, int64_t blockCount;
                                     int64_t dstStride = 0, int64_t srcStride = 0){
        
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.dstStride = dstStride;
        dataCopyParams.srcStride = srcStride;

        DataCopy(dst, src, dataCopyParams);
    }

    /* @brief: 搬出数据
    * Dst tensor, Src tensor
    */
    template <typename T = uint16_t>
    __aicore__ inline DataCopyOutCustom(GlobalTensor<T>& dst, LocalTensor<T>& src, 
                                     int64_t dstOffset, int64_t srcOffset, 
                                     int64_t blockLen, int64_t blockNum;
                                     int64_t dstStride = 0, int64_t srcStride = 0){
        
        DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = blockCount;
        dataCopyParams.blockLen = blockLen;
        dataCopyParams.dstStride = dstStride;
        dataCopyParams.srcStride = srcStride;

        DataCopy(dst, src, dataCopyParams);
    }


    /* @brief: 对[M, N]的向量进行除法
    * input: srcTensor, M, N, scalar, axis
    * output: dstTensor
    */
    template <typename T = uint16_t>
    __aicore__ inline void TensorDiv(){
        // 扩充
    }


    /* @brief: 处理对qhash的累加
    * input: qHashRowtensor, qHash tensor, qHash [Dim1Len, Dim2Len], axis, curTile, 
    */
    template <typename T = uint16_t>
    __aicore__ inline void QHashGroup(LocalTensor<T>& outputRowtensor, LocalTensor<T>& qHash, 
                                      uint32_t Dim1Len, uint32_t Dim2Len, bool axis, uint32_t curTile){
        
        for (uint32_t i = 0; i < param_.hDimTilingNum; i++){

            auto tileLength = i == param_.hDimTilingNum - 1 ? param_.hDimTilingTailLen : param_.hDimTilingLen;   
            auto groupLenth = param_.groupNum;
            
            int64_t dstOffset;
            int64_t srcOffset;
            int64_t blockLen;
            int64_t blockNum;

            // load qHash
            DataCopyCustom(qHash, qHashGm, dstOffset, srcOffset, blockLen, blockNum);

            CumSumConfig config;
            config.isLastAxis = asix;
            config.isReuseSource = true;
            config.outputLastRow = true;

            CumSumInfo cumSumInfo;
            cumSumInfo.outter = Dim1Len;
            cumSumInfo.inner = Dim2Len;

            CumSum<config>(qHash, outputRowtensor, qHash, cumSumInfo);
        }
    }

    /* @brief: 计算kHash和qHash的距离，通过求XOR和右移看奇偶获取汉明距离
    * input: qHashGroup, kHash, kHash [Dim1Len, Dim2Len], curTile
    * output: hammingSumUB
    */
    template <typename T = uint16_t>
    __aicore__ inline void Hamming(LocalTensor<T>& hamming, LocalTensor<T>& qHashGroup, LocalTensor<T>& kHash,
                                   uint32_t Dim1Len, uint32_t curTile){
        
        for (uint32_t i = 0; i < param_.hDimTilingNum; i++){
            
            auto tileLength = i == param_.hDimTilingNum - 1 ? param_.hDimTilingTailLen : param_.hDimTilingLen;   
            auto seqLen = Dim1Len;

            int64_t dstOffset;
            int64_t srcOffset;
            int64_t blockLen;
            int64_t blockNum;
            
            // load kHashTile
            DataCopyCustom(kHash, kHashGm, dstOffset, srcOffset, blockLen, blockNum);

            // compute Hamming
            XOR(XORUB, );

            for (uint32_t j = 0; j < sizeof(hashDataType); j++){
                ShiftRight(XORUB, 1, )
            }

        }


    }

    /* @brief: 在这里对输入的序列进行Chunk缩减
    * input: inTensor; ChunkSize, ChunkMode
    * output: outTensor
    */
    template <typename T = uint16_t>
    __aicore__ inline void ChunkCompress(LocalTensor<T>& outTensor, LocalTensor<T>& inTensor, uint32_t chunkSize, uint32_t chunkMode){
        
    }

    /* @brief: 在这里计算topk
    * input: srcValueLocal, srcIndexLocal, k, 
    * output: dstValueLocal, dstIndexLocal
    */
    template <typename T>
    __aicore__ inline void TopKCustom(const LocalTensor<T> &dstValueLocal, const LocalTensor<int32_t> &dstIndexLocal,
        const LocalTensor<T> &srcValueLocal, const LocalTensor<int32_t> &srcIndexLocal, const int32_t k, 
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
        // VECALC
        int64_t XORBaseUB = 0;
        int64_t rightShiftBaseUB = XORBaseUB + param_.hammingGroupTilingSize;
        int64_t hammingGroupBaseUB = rightShiftBaseUB + param_.hammingGroupTilingSize;
        int64_t hammingSumBaseUB = hammingGroupBaseUB + param_.hammingGroupTilingSize;
        int64_t hammingChunkBaseUB = hammingSumBaseUB + param_.hammingSumTilingSize;
        // VECOUT
        int64_t indexChunkBaseUB = 0;
        int64_t topKChunkBaseUB = indexChunkBaseUB + param_.indexChunkTilingSize;

        for (uint32_t core_idx = GetBlockIdx(); core_idx < totalNum; core_idx += blocknum){
            
            LocalTensor<int16_t> qHashUB, kHashUB, // input
                                 XORUB, rightShiftUB, // hamming intermediate
                                 hammingGroupUB, hammingSumUB, hammingChunkUB, // hamming result 
                                 indexChunkUB, topKChunkUB; // result
            
            // 此处设置的时候就需要考虑 multi buffer
            // VECIN
            SetTensorAddr<hashDataType>(qHashUB, param_.qHashTilingSize, qHashBaseUB, 9);
            SetTensorAddr<hashDataType>(kHashUB, param_.kHashTilingSize, kHashBaseUB, 9);
            // VECALC
            SetTensorAddr<hashDataType>(XORUB, param_.hammingGroupTilingSize, XORBaseUB, 11);
            SetTensorAddr<hashDataType>(rightShiftUB, param_.hammingGroupTilingSize, rightShiftBaseUB, 11);
            SetTensorAddr<hashDataType>(hammingGroupUB, param_.hammingGroupTilingSize, hammingGroupBaseUB, 11);
            SetTensorAddr<hashDataType>(hammingSumUB, param_.hammingSumTilingSize, hammingSumBaseUB, 11);
            SetTensorAddr<hashDataType>(hammingChunkUB, param_.hammingChunkTilingSize, hammingChunkBaseUB, 11); 
            // VECOUT
            SetTensorAddr<hashDataType>(indexChunkUB, param_.indexChunkTilingSize, indexChunkBaseUB, 10);
            SetTensorAddr<hashDataType>(topKChunkUB, param_.topKChunkTilingSize, topKChunkBaseUB, 10);
            
            // auto kHashOffset = loop_ping_flag ? kHashBaseUB + kHashSingleTilingSize : kHashBaseUB;
            // VECIN
            auto qHashUBOffset = loop_ping_flag ? qHashBaseUB + param_.qHashSingleTilingSize : qHashBaseUB;
            auto kHashUBOffset = loop_ping_flag ? kHashBaseUB + param_.kHashSingleTilingSize : kHashBaseUB;
            // VECALC
            auto XORUBOffset = loop_ping_flag ? XORBaseUB + param_.hammingGroupSingleTilingSize : XORBaseUB;
            auto rightShiftUB = loop_ping_flag ? rightShiftBaseUB + param_.hammingGroupSingleTilingSize : rightShiftBaseUB;
            auto hammingGroupUBOffset = loop_ping_flag ? hammingGroupBaseUB + param_.hammingGroupSingleTilingSize : hammingGroupBaseUB;
            auto hammingSumUBOffset = loop_ping_flag ? hammingSumBaseUB + param_.hammingSumSingleTilingSize : hammingSumBaseUB;
            auto hammingChunkUBOffset = loop_ping_flag ? hammingChunkBaseUB + param_.hammingChunkSingleTilingSize : hammingChunkBaseUB;
            // VECOUT
            auto indexChunkUBOffset = loop_ping_flag ? indexChunkBaseUB + param_.indexChunkSingleTilingSize : indexChunkBaseUB;
            auto topKChunkUBOffset = loop_ping_flag ? topKChunkUB + param_.topkChunkSingleTilingSize : topKChunkUB;

            // 先计算Hamming dist
            for (uint32_t hDimTiling = 0; hDimTiling < hDimTilingNum; hDimTiling += 1){
                // 计算tile大小
                auto curHDim = hDimTiling == hDimTilingNum - 1 ? param_.hDimTilingTailLen : param_.hDimTilingLen;
                
                // load qHash and compute qhashgroup
                QHashGroup(qHashGroupUB, qHashUB, param_.groupNum, curHDim, false, hDimTiling); // cumsum
            }

            // // 一次性算完全部qHash的group之后，保存group，此时可以复用qHash的内存空间
            // qHashUB.Free();
            // SetTensorAddr<hashDataType>(kHashUB, param_.kHashTilingSize, kHashBaseUB, 9);
            // auto kHashOffset = loop_ping_flag ? kHashBaseUB + kHashSingleTilingSize : kHashBaseUB;

            // 再计算汉明距离
            for (uint32_t seqLenTiling = 0; seqLenTiling < seqLenTilingNum; seqLenTiling += 1){
                // 计算tile大小
                auto curSeqLen = seqLenTiling == seqLenTilingNum - 1 ? param_.seqLenTilingTailLen : param_.seqLenTilingLen;

                Hamming(hammingSumUB, qHashGroupUB, kHash, curSeqLen, seqLenTiling); // [1, T_S]
                
                ChunkCompress(hammingChunkUB, hammingSumUB, param_.chunkSize, param_.chunkMode); // [1, chunkNum]

                GenerateIndex(); // ArithProgression
                TopKCustom(); // [1, compressTopK]
            }

            // get output GM offset
            int64_t dstOffset;
            int64_t srcOffset;
            int64_t blockLen;
            int64_t blockNum;

            DataCopyOutCustom(indexGm, indexChunkUB, dstOffset, srcOffset, blockLen, blockNum);

            loop_ping_flag = 1 - loop_ping_flag;
        }
    }

private:

    TilingParam param_;
    
    // TPipe pipe;
    // TQue<TPosition::VECIN, BUFFER_NUM> inQueueQHash;
    // TQue<TPosition::VECIN, BUFFER_NUM> inQueueKHash;
    // TQue<TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    // TBuf<TPosition::VECCALC> tmpBuf2;

    GlobalTensor<hashDataType> qHashGm;
    GlobalTensor<hashDataType> kHashGm;
    GlobalTensor<hashDataType> indexGm;

    // uint32_t coef;
    // uint32_t tileNum;
    // uint32_t tileLength;
    // // uint32_t 
    // uint32_t lastTileLength;
};

extern "C" __global__ __aicore__ void hamming_dist_top_k_custom(GM_ADDR qHash, GM_ADDR kHash, GM_ADDR index, HammingTilingData tiling)
{
    int32_t bufferNum = tiling.bufferNum;
    assert(bufferNum <= 2);

    KernelHammingDistTopK<uint16_t> op;
    op.Init<bufferNum>(qHash, kHash, index, tiling);
    op.Process();

}
