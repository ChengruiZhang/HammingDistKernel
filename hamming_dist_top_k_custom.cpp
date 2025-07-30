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
        GM_ADDR qHash_ = qHash;
        GM_ADDR kHash_ = kHash;

        auto blockidx = AscendC::GetBlockIdx();
        qHashGm.SetGlobalBuffer();
        kHashGm.SetGlobalBuffer();
        indexGm.SetGlobalBuffer();

        // parse tiling factor
        this->tileNum = tiling.tileNum; // 此处的tiling是考虑包括
        this->tileLength = tiling.tileLength / BUFFER_NUM;
        
        // AscendC::PRINTF("L682");
        if (qHash)
            return;
        if (kHash)
            return;
        if (index)
            return;
        (void)tiling;

    }

    /*
    * @brief: 获取每一个核的偏移：对于GM的qkhash读取位置，以及写入index的位置；
    * 此外，计算出当前核对应的输出大小MLen、NLen
    */
    __aicore__ inline void GetCoreOffset(int64_t coreidx_){
        qHashCoreOffset;
        kHashCoreOffset;
        indexCoreOffset;
    }

    /* @brief:
    * 第一版的分块策略为，针对每一个核处理的qHash[G, HDim] kHash[SeqLen, HDim]，在Seqlen维度和HDim维度进行分块，不对qhash的G进行分块
    * 
    */
    __aicore__ inline void Process()
    {
        uint8_t blocknum = AscendC::GetBlockNum();
        uint32_t loop_idx = AscendC::GetBlockIdx();

        int32_t loop_ping_flag = 0;
        
        LocalTensor<int32_t> qHashUB, qHashGroupUB, kHashUB, indexHashUB;
        AscendC::SetTensorAddr<>(localtensor, size, base, Tpos); // 此处设置的时候就需要考虑double buffer
        
        for (uint32_t loop_idx = GetBlockIdx(); loop_idx < totalNum; loop_idx += blocknum){
            
            GetCoreOffset(loop_idx);

            auto qHashUB = loop_ping_flag ? L0C_base + L0C_PINGPONG_BUFFER_LEN : L0C_base;
            auto kHashUB = loop_ping_flag ? L0C_base + L0C_PINGPONG_BUFFER_LEN : L0C_base;
            auto indexHashUB = loop_ping_flag ? L0C_base + L0C_PINGPONG_BUFFER_LEN : L0C_base;

            for (uint32_t seqLenTiling = 0; seqLenTiling < seqLenTilingNum; seqLenTiling += 1){
                for (uint32_t hDimTiling = 0; hDimTiling < hDimTilingNum; hDimTiling += 1){
                    QHashGroup(); // cumsum
                    Hamming(); // [1, T_S]
                }
                ConCat(); // [1, S]
                Block(); // [1, chunkNum]
                TopK(); // [1, compressTopK]
            }
            CopyOut();
        }
    }

private:

    TilingParam param_;
    
    // before
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueQHash;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueKHash;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<hashDataType> xGm;
    AscendC::GlobalTensor<hashDataType> yGm;
    AscendC::GlobalTensor<hashDataType> zGm;

    uint32_t coef;
    uint32_t tileNum;
    uint32_t tileLength;
    // uint32_t 
    uint32_t lastTileLength;
};

extern "C" __global__ __aicore__ void hamming_dist_top_k_custom(GM_ADDR qHash, GM_ADDR kHash, GM_ADDR index, HammingTilingData tiling)
{
    int32_t bufferNum = tiling.bufferNum;
    assert(bufferNum <= 2);

    KernelHammingDistTopK<uint16_t> op;
    op.Init<bufferNum>(qHash, kHash, index, tiling);
    op.Process();

}
