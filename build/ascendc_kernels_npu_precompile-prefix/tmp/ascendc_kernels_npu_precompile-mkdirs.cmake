# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/westhpc/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/device_precompile_project")
  file(MAKE_DIRECTORY "/home/westhpc/Ascend/ascend-toolkit/latest/compiler/tikcpp/ascendc_kernel_cmake/device_precompile_project")
endif()
file(MAKE_DIRECTORY
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-build"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix/tmp"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix/src"
  "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/westhpc/RayCode/hamming_dist_top_k/HammingDistKernel/build/ascendc_kernels_npu_precompile-prefix/src/ascendc_kernels_npu_precompile-stamp${cfgdir}") # cfgdir has leading slash
endif()
