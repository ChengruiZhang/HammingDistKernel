#ifndef __TOPK_CUSTOM__KERNEL_FUN_H__
#define __TOPK_CUSTOM__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define topk_custom topk_custom_origin
#include "/home/westhpc/RayCode/hamming_dist_top_k/hamming_final_2/kernel_launch_method_by_direct/topk_custom.cpp"

#undef topk_custom
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_topk_custom_kernel(
__attribute__((cce_global)) uint8_t* qHash, __attribute__((cce_global)) uint8_t* kHash, __attribute__((cce_global)) uint8_t* topK, __attribute__((cce_global)) uint8_t* workspace, __attribute__((cce_global)) uint8_t* tiling, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = workspace;
#else
    workspace_param = tiling;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    workspace = workspace_usr;
#else
    tiling = workspace_usr;
#endif
#endif
    topk_custom_origin(qHash, kHash, topK, workspace, tiling);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
