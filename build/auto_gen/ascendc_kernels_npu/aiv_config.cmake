set(MIX_SOURCES
)
set(AIV_SOURCES
    /data/RayCode/hamming_dist_top_k/HammingDistKernel/build/auto_gen/ascendc_kernels_npu/auto_gen_hamming_dist_top_k_custom.cpp
)
set_source_files_properties(/data/RayCode/hamming_dist_top_k/HammingDistKernel/build/auto_gen/ascendc_kernels_npu/auto_gen_hamming_dist_top_k_custom.cpp
    PROPERTIES COMPILE_DEFINITIONS ";auto_gen_hamming_dist_top_k_custom_kernel=hamming_dist_top_k_custom_0;ONE_CORE_DUMP_SIZE=1048576"
)
