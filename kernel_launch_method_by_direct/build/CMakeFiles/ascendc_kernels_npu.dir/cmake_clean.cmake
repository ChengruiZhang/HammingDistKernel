file(REMOVE_RECURSE
  "lib/libascendc_kernels_npu.a"
  "lib/libascendc_kernels_npu.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/ascendc_kernels_npu.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
