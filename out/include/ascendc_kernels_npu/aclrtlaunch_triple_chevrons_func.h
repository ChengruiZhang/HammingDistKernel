
#ifndef HEADER_ACLRTLAUNCH_HAMMING_DIST_TOP_K_CUSTOM_HKERNEL_H_
#define HEADER_ACLRTLAUNCH_HAMMING_DIST_TOP_K_CUSTOM_HKERNEL_H_


struct HammingTilingData;


extern "C" uint32_t aclrtlaunch_hamming_dist_top_k_custom(uint32_t blockDim, void* stream, void* x, void* y, void* z, HammingTilingData* tiling);

inline uint32_t hamming_dist_top_k_custom(uint32_t blockDim, void* hold, void* stream, void* x, void* y, void* z, HammingTilingData* tiling)
{
    (void)hold;
    return aclrtlaunch_hamming_dist_top_k_custom(blockDim, stream, x, y, z, tiling);
}

#endif
