# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/host_project"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-build"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix/tmp"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix/src"
  "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/data/home/2301111796/HammingDistTiling/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp${cfgdir}") # cfgdir has leading slash
endif()
