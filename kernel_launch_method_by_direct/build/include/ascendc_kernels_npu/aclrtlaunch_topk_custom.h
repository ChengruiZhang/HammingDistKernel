#ifndef HEADER_ACLRTLAUNCH_TOPK_CUSTOM_H
#define HEADER_ACLRTLAUNCH_TOPK_CUSTOM_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_topk_custom(uint32_t blockDim, aclrtStream stream, void* qHash, void* kHash, void* topK, void* workspace, void* tiling);
#endif
