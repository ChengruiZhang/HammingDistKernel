#ifndef HEADER_ACLRTLAUNCH_HAMMING_DIST_TOP_K_CUSTOM_H
#define HEADER_ACLRTLAUNCH_HAMMING_DIST_TOP_K_CUSTOM_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_hamming_dist_top_k_custom(uint32_t blockDim, aclrtStream stream, void* qHash, void* kHash, void* index, HammingTilingData* tiling);
#endif
