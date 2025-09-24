add_library(ascendc_runtime_obj OBJECT IMPORTED)
set_target_properties(ascendc_runtime_obj PROPERTIES
    IMPORTED_OBJECTS "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/elf_tool.c.o;/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_runtime.cpp.o"
)
