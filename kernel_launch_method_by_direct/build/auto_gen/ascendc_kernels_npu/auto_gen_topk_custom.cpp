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
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
GM_ADDR dumpAddr,
#endif
__attribute__((cce_global)) uint8_t* qHash, __attribute__((cce_global)) uint8_t* kHash, __attribute__((cce_global)) uint8_t* topK, __attribute__((cce_global)) uint8_t* workspace, __attribute__((cce_global)) uint8_t* tiling, GM_ADDR overflow_status) {
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    AscendC::InitDump(false, dumpAddr, ONE_CORE_DUMP_SIZE);
#ifdef ASCENDC_TIME_STAMP_ON
    AscendC::PrintTimeStamp(static_cast<uint32_t>(AscendC::TimeStampId::TIME_STAMP_WRAP_INIT_DUMP));
#endif
#endif

#ifdef ASCENDC_DUMP
    uint64_t __ascendc_tStamp = 0;
    uint64_t __ascendc_version = 0;
     __gm__ char* __ascendc_versionStr = nullptr;
    GetCannVersion(__ascendc_versionStr, __ascendc_version, __ascendc_tStamp);
    if (__ascendc_tStamp == 0) {
        AscendC::printf("[WARNING]: CANN TimeStamp is invalid, CANN TimeStamp is %u\n", __ascendc_tStamp);
    } else {
        AscendC::printf("CANN Version: %s, TimeStamp: %u\n", (__gm__ const char*)(__ascendc_versionStr), __ascendc_tStamp);
    }
#endif
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
