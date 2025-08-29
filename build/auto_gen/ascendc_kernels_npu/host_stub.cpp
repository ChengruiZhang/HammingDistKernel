#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <dlfcn.h>
#include <securec.h>

#ifndef ASCENDC_DUMP
#define ASCENDC_DUMP 1
#endif

#if defined(ASCENDC_DUMP) && (ASCENDC_DUMP == 0)
    #undef ASCENDC_DUMP
#endif

#ifdef ASCENDC_DUMP
#define ASCENDC_EXCEPTION_DUMP_HEAD 2U

typedef struct rtArgsSizeInfo {
    void *infoAddr;
    uint32_t atomicIndex;
} rtArgsSizeInfo_t;
#endif

static char ascendcErrMsg[1024] = {0};

static void *g_kernel_handle_aiv = nullptr;

struct ascend_kernels {
    uint32_t version;
    uint32_t type_cnt;
    uint32_t aiv_type;
    uint32_t aiv_len;
    uint32_t aiv_file_len;
    uint8_t aiv_buf[110592];
} __ascend_kernel_ascend910b3_ascendc_kernels_npu __attribute__ ((section (".ascend.kernel.ascend910b3.ascendc_kernels_npu"))) = {1,1,1,110592,110592,{0}};

extern "C" {
uint32_t RegisterAscendBinary(const char *fileBuf, size_t fileSize, uint32_t type, void **handle);
uint32_t LaunchAscendKernel(void *handle, const uint64_t key, const uint32_t blockDim, void **args,
                            uint32_t size, const void *stream);
uint32_t GetAscendCoreSyncAddr(void **addr);
int UnregisterAscendBinary(void *hdl);
void StartAscendProf(const char *name, uint64_t *startTime);
void ReportAscendProf(const char *name, uint32_t blockDim, uint32_t taskType, const uint64_t startTime);
bool GetAscendProfStatus();
uint32_t AllocAscendMemDevice(void **devMem, uint64_t size);
uint32_t FreeAscendMemDevice(void *devMem);
bool AscendCheckSoCVersion(const char *socVersion, char* errMsg);
void AscendProfRegister();
uint32_t GetCoreNumForMixVectorCore(uint32_t *aiCoreNum, uint32_t *vectorCoreNum);
uint32_t LaunchAscendKernelForVectorCore(const char *opType, void *handle, const uint64_t key, void **args, uint32_t size,
    const void *stream, bool enbaleProf, uint32_t aicBlockDim, uint32_t aivBlockDim, uint32_t aivBlockDimOffset);
int32_t rtSetExceptionExtInfo(const rtArgsSizeInfo_t * const sizeInfo);

namespace Adx {
    void *AdumpGetSizeInfoAddr(uint32_t space, uint32_t &atomicIndex);
}
}
namespace Adx {

    void AdumpPrintWorkSpace(const void *workSpaceAddr, const size_t dumpWorkSpaceSize,
                            void *stream, const char *opType);
}

    class KernelHandleGradUnregister {
    private:
        KernelHandleGradUnregister() {}

    public:
        KernelHandleGradUnregister(const KernelHandleGradUnregister&) = delete;
        KernelHandleGradUnregister& operator=(const KernelHandleGradUnregister&) = delete;

        static KernelHandleGradUnregister& GetInstance() {
            static KernelHandleGradUnregister instance;
            return instance;
        }
        ~KernelHandleGradUnregister(){
            if (g_kernel_handle_aiv) {
                UnregisterAscendBinary(g_kernel_handle_aiv);
                g_kernel_handle_aiv = nullptr;
            }
        }
    };

static void __register_kernels(void) __attribute__((constructor));
void __register_kernels(void)
{
    const char* compileSocVersion = "ascend910b3";
    uint32_t ret;

    bool checkSocVersion = AscendCheckSoCVersion(compileSocVersion, ascendcErrMsg);
    if (!checkSocVersion) {
        return;
    }
    ret = RegisterAscendBinary(
        (const char *)__ascend_kernel_ascend910b3_ascendc_kernels_npu.aiv_buf,
        __ascend_kernel_ascend910b3_ascendc_kernels_npu.aiv_file_len,
        1,
        &g_kernel_handle_aiv);
    if (ret != 0) {
        printf("RegisterAscendBinary aiv ret %u \n", ret);
    }

    AscendProfRegister();
}

#ifdef ASCENDC_DUMP
static void ascendc_set_exception_dump_info(uint32_t dumpSize)
{
    uint32_t atomicIndex = 0U;
    uint32_t addrNum = 1U;
    void *exceptionDumpAddr = Adx::AdumpGetSizeInfoAddr(addrNum + ASCENDC_EXCEPTION_DUMP_HEAD, atomicIndex);
    if (exceptionDumpAddr == nullptr) {
        printf("Get exceptionDumpAddr is nullptr.\n");
        return;
    }


    uint64_t *sizeInfoAddr = reinterpret_cast<uint64_t *>(exceptionDumpAddr);
    *sizeInfoAddr = static_cast<uint64_t>(atomicIndex);
    sizeInfoAddr++;

    *sizeInfoAddr = static_cast<uint64_t>(1);
    sizeInfoAddr++;

    *sizeInfoAddr = dumpSize * 75;
    constexpr uint64_t workspaceOffset = (4ULL << 56ULL);
    *sizeInfoAddr |= workspaceOffset;

    const rtArgsSizeInfo sizeInfo = {exceptionDumpAddr, atomicIndex};
    int32_t ret = rtSetExceptionExtInfo(&sizeInfo);
    if (ret != 0) {
        printf("rtSetExceptionExtInfo failed, ret = %d.\n", ret);
    }
}
#endif


struct HammingTilingData {


    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t seqLenPad;
    uint32_t seqBlock;

    uint32_t topK;
    uint32_t topKCompressed;
    uint32_t topKComprssedPad;

    uint32_t hidDim;
    uint32_t hidDimCompressNum;
    uint32_t hidDimCompressPadNum;
    uint32_t hidDimCompressAddNum;

    uint32_t totalNum;
    uint32_t groupNum;
    uint32_t bufferNum;

    uint32_t scalarSize;



    uint32_t qHashCoreOffset;
    uint32_t kHashCoreOffset;
    uint32_t indexCoreOffset;
    uint32_t qHashCoreOffsetBlock;
    uint32_t kHashCoreOffsetBlock;
    uint32_t indexCoreOffsetBlock;


    uint32_t seqLenTilingLen;
    uint32_t seqLenTilingNum;
    uint32_t seqLenTilingTailLen;
    uint32_t seqLenBlockNum;


    uint32_t qHashTilingSize;
    uint32_t qHashSingleTilingSize;
    uint32_t kHashTilingSize;
    uint32_t kHashSingleTilingSize;

    uint32_t tmpWorkSpaceSize;


    uint32_t reduceSumWorkSpaceSize = 512;

    uint32_t hammingXORTilingSize;
    uint32_t hammingXORSingleTilingSize;
    uint32_t hammingRightTilingSize;
    uint32_t hammingRightSingleTilingSize;
    uint32_t hammingCastTilingSize;
    uint32_t hammingCastSingleTilingSize;
    uint32_t hammingLastRowTilingSize;
    uint32_t hammingLastRowSingleTilingSize;
    uint32_t hammingSumTilingSize;
    uint32_t hammingSumSingleTilingSize;
    uint32_t hammingCumTilingSize;
    uint32_t hammingCumSingleTilingSize;
    uint32_t hammingReduceTilingSize;
    uint32_t hammingReduceSingleTilingSize;
    uint32_t hammingResultTilingSize;
    uint32_t hammingResultSingleTilingSize;

    uint32_t resultSize;
    uint32_t resultSingleSize;

    uint32_t resultChunkSize;
    uint32_t resultChunkSingleSize;


    uint32_t chunkSize;
    uint32_t chunkRepeat;
    uint32_t chunkTailMask;
    uint32_t chunkMode;
    uint32_t chunkTopKNum;


    uint32_t indexChunkSize;
    uint32_t indexChunkSingleSize;
    uint32_t topKChunkSize;
    uint32_t topKChunkSingleSize;
# 113 "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/hamming_dist_top_k_custom_tiling.h"
};



uint32_t launch_and_profiling_hamming_dist_top_k_custom(uint64_t func_key, uint32_t blockDim, void* stream, void **args, uint32_t size)
{
    uint64_t startTime;
    const char *name = "hamming_dist_top_k_custom";
    bool profStatus = GetAscendProfStatus();
    if (profStatus) {
        StartAscendProf(name, &startTime);
    }
    if (g_kernel_handle_aiv == nullptr) {
        printf("[ERROR] %s\n", ascendcErrMsg);
        return 0;
    }
    uint32_t ret = LaunchAscendKernel(g_kernel_handle_aiv, func_key, blockDim, args, size, stream);
    if (ret != 0) {
        printf("LaunchAscendKernel ret %u\n", ret);
    }
    if (profStatus) {
        ReportAscendProf(name, blockDim, 1, startTime);
    }
    return ret;
}

extern "C" uint32_t aclrtlaunch_hamming_dist_top_k_custom(uint32_t blockDim, void* stream, void* qHash, void* kHash, void* index, HammingTilingData* tiling)
{
    struct {
    #if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
            void* __ascendc_dump;
    #endif
        alignas(((alignof(void*) + 3) >> 2) << 2) void* qHash;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* kHash;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* index;
        alignas(((alignof(HammingTilingData) + 3) >> 2) << 2) HammingTilingData tiling;
        alignas(((alignof(void*) + 3) >> 2) << 2) void* __ascendc_overflow;
    } __ascendc_args;

    uint32_t __ascendc_ret;
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    constexpr uint32_t __ascendc_one_core_dump_size = 1048576;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_dump), __ascendc_one_core_dump_size * 75);
#endif
    constexpr uint32_t __ascendc_overflow_status_size = 8;
    AllocAscendMemDevice(&(__ascendc_args.__ascendc_overflow), __ascendc_overflow_status_size);
    __ascendc_args.qHash = qHash;
    __ascendc_args.kHash = kHash;
    __ascendc_args.index = index;
    (void) memcpy_s(&__ascendc_args.tiling, sizeof(__ascendc_args.tiling), tiling, sizeof(__ascendc_args.tiling));

    const char *__ascendc_name = "hamming_dist_top_k_custom";
    ascendc_set_exception_dump_info(__ascendc_one_core_dump_size);
    __ascendc_ret = launch_and_profiling_hamming_dist_top_k_custom(0, blockDim, stream, (void **)&__ascendc_args, sizeof(__ascendc_args));
    KernelHandleGradUnregister::GetInstance();
#if defined ASCENDC_DUMP || defined ASCENDC_TIME_STAMP_ON
    Adx::AdumpPrintWorkSpace(__ascendc_args.__ascendc_dump, __ascendc_one_core_dump_size * 75, stream, __ascendc_name);
    FreeAscendMemDevice(__ascendc_args.__ascendc_dump);
#endif
    FreeAscendMemDevice(__ascendc_args.__ascendc_overflow);
    return __ascendc_ret;
}
