# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/device_precompile_project"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-build"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix/tmp"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix/src"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp${cfgdir}") # cfgdir has leading slash
endif()
