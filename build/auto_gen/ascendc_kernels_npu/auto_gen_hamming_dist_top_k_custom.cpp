#ifndef __HAMMING_DIST_TOP_K_CUSTOM__KERNEL_FUN_H__
#define __HAMMING_DIST_TOP_K_CUSTOM__KERNEL_FUN_H__

#undef __global__
#define __global__ inline
#define hamming_dist_top_k_custom hamming_dist_top_k_custom_origin
#include "/data/home/2301111796/HammingDistTiling/hamming_dist_top_k_custom.cpp"

#undef hamming_dist_top_k_custom
#undef __global__
#if ASCENDC_CPU_DEBUG
#define __global__
#else
#define __global__ __attribute__((cce_kernel))
#endif

#ifndef ONE_CORE_DUMP_SIZE
#define ONE_CORE_DUMP_SIZE 1048576 * 1
#endif

extern "C" __global__ [aicore] void auto_gen_hamming_dist_top_k_custom_kernel(
__attribute__((cce_global)) uint8_t* x, __attribute__((cce_global)) uint8_t* y, __attribute__((cce_global)) uint8_t* z, HammingTilingData tiling, GM_ADDR overflow_status) {
#if defined(HAVE_WORKSPACE)
    GM_ADDR workspace_param;
    GM_ADDR workspace_usr;
#if defined(HAVE_TILING)
    workspace_param = z;
#else
    workspace_param = tiling;
#endif
    AscendC::SetSysWorkspaceForce(workspace_param);
    workspace_usr = AscendC::GetUserWorkspace(workspace_param);
#if defined(HAVE_TILING)
    z = workspace_usr;
#else
    tiling = workspace_usr;
#endif
#endif
    hamming_dist_top_k_custom_origin(x, y, z, tiling);
#if defined(ASCENDC_DUMP) && defined(ASCENDC_DEBUG)
    AscendC::WriteBackOverflow(overflow_status);
#endif
}

#endif
#include "inner_interface/inner_kernel_operator_intf.h"
