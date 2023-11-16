cmake_minimum_required(VERSION 3.26)

project(FMHADecoderMain LANGUAGES CXX)

message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER} (need hipcc)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(exe_name attention_forward_decoder_main)
set(project_root_dir /xformers)
set(xformers_csrc ${project_root_dir}/xformers/csrc)
set(sources ${xformers_csrc}/attention/hip_fmha/attention_forward_decoder.hip)

set(ck_include ${project_root_dir}/third_party/composable_kernel/include/)
set(torch_include /opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/include)

set_source_files_properties(${sources} PROPERTIES LANGUAGE CXX)
add_executable(${exe_name} ${sources})

message("sources: ${sources}")

message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")
find_package(HIP REQUIRED)
find_package(ROCM REQUIRED PATHS /opt/rocm)
include(ROCMInstallTargets)

message("HIP_VERSION: ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}")

set_target_properties(${exe_name} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${exe_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_compile_options(${exe_name} PUBLIC 
  -O3 
  --offload-arch=${GPU_TARGETS}
  -fno-gpu-rdc)

target_include_directories(${exe_name} PUBLIC 
  ${xformers_csrc} 
  ${xformers_csrc}/attention/hip_fmha
  ${ck_include}
  ${ck_include}/ck/tensor_operation/gpu/device
  ${ck_include}/ck/tensor_operation/gpu/device/impl
  ${ck_include}/ck/tensor_operation/gpu/element
  ${torch_include}
  ${torch_include}/torch/csrc/api/include
  ${torch_include}/TH
  ${torch_include}/THC
  ${torch_include}/THH
)

target_link_directories(${exe_name} PUBLIC
  /opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/lib
  /opt/conda/envs/py_3.8/lib
  /opt/rocm/lib
  /opt/rocm/hip/lib
)

target_link_libraries(${exe_name} PUBLIC
  c10
  c10_hip
  torch
  torch_python
  torch_hip
  torch_cpu
  python3.8
  amdhip64
)

target_compile_definitions(${exe_name} PUBLIC 
  ATTN_FWD_DECODER_MAIN=1
  GLIBCXX_USE_CXX11_ABI=1
  __HIP_PLATFORM_HCC__=1
  USE_ROCM=1
)

include(CMakePrintHelpers)
cmake_print_properties(TARGETS ${exe_name} PROPERTIES LINK_LIBRARIES LINK_DIRECTORIES INCLUDE_DIRECTORIES COMPILE_DEFINITIONS COMPILE_OPTIONS)

rocm_install(TARGETS ${exe_name})