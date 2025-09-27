
#ifndef HEADER_ACLRTLAUNCH_TOPK_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_TOPK_CUSTOM_HKERNEL_H_



extern "C" uint32_t aclrtlaunch_topk_custom(uint32_t blockDim, void* stream, void* qHash, void* kHash, void* topK, void* workspace, void* tiling);

inline uint32_t topk_custom(uint32_t blockDim, void* hold, void* stream, void* qHash, void* kHash, void* topK, void* workspace, void* tiling)
{
    (void)hold;
    return aclrtlaunch_topk_custom(blockDim, stream, qHash, kHash, topK, workspace, tiling);
}

#endif
