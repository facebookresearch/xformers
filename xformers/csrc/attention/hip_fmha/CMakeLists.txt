cmake_minimum_required(VERSION 3.26)

project(FMHA-Decoder-Main)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(project_root_dir /xformers)
set(xformers_csrc ${project_root_dir}/xformers/csrc)
set(sources ${xformers_csrc}/attention/hip_fmha/attention_forward_decoder.hip)

set(ck_include ${project_root_dir}/third_party/composable_kernel/include/ck)
set(torch_include /opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/include)

set(CMAKE_CXX_COMPILER /opt/rocm/hip/bin/hipcc)
set(CMAKE_CXX_LINK_EXECUTABLE /opt/rocm/hip/bin/hipcc)

add_executable(attention_forward_decoder_main ${sources})
message("CMAKE_CXX_COMPILER_ID: ${CMAKE_CXX_COMPILER_ID}")

find_package(HIP REQUIRED)

message("HIP_VERSION: ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}")

set_target_properties(attention_forward_decoder_main PROPERTIES LINKER_LANGUAGE CXX)

target_compile_options(attention_forward_decoder_main PUBLIC 
  -fPIC 
  -O3 
  --offload-arch=gfx90a
  -fno-gpu-rdc)

target_include_directories(attention_forward_decoder_main PUBLIC 
  ${xformers_csrc} 
  ${xformers_csrc}/attention/hip_fmha
  ${ck_include}/tensor_operation/gpu/device
  ${ck_include}/tensor_operation/gpu/device/impl
  ${ck_include}/tensor_operation/gpu/element
  ${torch_include}
  ${torch_include}/torch/csrc/api/include
  ${torch_include}/TH
  ${torch_include}/THC
  ${torch_include}/THH
)

target_link_directories(attention_forward_decoder_main PUBLIC
  /opt/conda/envs/py_3.8/lib/python3.8/site-packages/torch/lib
  /opt/conda/envs/py_3.8/lib
  /opt/rocm/lib
  /opt/rocm/hip/lib
)

target_link_libraries(attention_forward_decoder_main PUBLIC
  c10
  c10_hip
  torch
  torch_python
  torch_hip
  torch_cpu
  python3.8
  amdhip64
)

target_compile_definitions(attention_forward_decoder_main PUBLIC 
  ATTN_FWD_DECODER_MAIN
)
