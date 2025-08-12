# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/host_project")
  file(MAKE_DIRECTORY "/usr/local/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/host_project")
endif()
file(MAKE_DIRECTORY
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-build"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix/tmp"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix/src"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_host-prefix/src/ascendc_kernels_npu_host-stamp${cfgdir}") # cfgdir has leading slash
endif()
