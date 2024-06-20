cmake_minimum_required(VERSION 3.26)

project(FMHADecoderMain LANGUAGES CXX HIP)

message("CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER} (need hipcc)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "-Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_VERBOSE_MAKEFILE on)

set(py_version 3.9)

set(exe_name attention_forward_decoder_main)
set(splitk_exe_name attention_forward_splitk_decoder_main)
set(project_root_dir /xformers)
set(xformers_csrc ${project_root_dir}/xformers/csrc)
set(sources ${xformers_csrc}/attention/hip_fmha/attention_forward_decoder.hip)
set(splitk_sources ${xformers_csrc}/attention/hip_fmha/attention_forward_splitk.hip)
set(ck_include ${project_root_dir}/third_party/composable_kernel_tiled/include/)
set(torch_include /opt/conda/envs/py_${py_version}/lib/python${py_version}/site-packages/torch/include)

set_source_files_properties(${sources} ${splitk_sources} PROPERTIES LANGUAGE HIP)
add_executable(${exe_name} ${sources})
add_executable(${splitk_exe_name} ${splitk_sources})

find_package(HIP REQUIRED)
find_package(ROCM REQUIRED PATHS /opt/rocm)
include(ROCMInstallTargets)

message("HIP_VERSION: ${HIP_VERSION_MAJOR}.${HIP_VERSION_MINOR}.${HIP_VERSION_PATCH}")

set_target_properties(${exe_name} ${splitk_exe_name} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${exe_name} ${splitk_exe_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${exe_name} ${splitk_exe_name} PROPERTIES HIP_ARCHITECTURES ${GPU_TARGETS})

target_compile_options(${exe_name} PUBLIC 
  -fno-gpu-rdc
  $<$<CONFIG:Debug>:
  --save-temps
  >
)

target_compile_options(${splitk_exe_name} PUBLIC 
  -fno-gpu-rdc
  $<$<CONFIG:Debug>:
  --save-temps
  -g
  -O0
  >
)

target_include_directories(${exe_name} PUBLIC 
  ${ck_include}                           # ck includes
  ${torch_include}                        # aten includes
  ${torch_include}/torch/csrc/api/include # torch includes
)

target_include_directories(${splitk_exe_name} PUBLIC 
  ${ck_include}                           # ck includes
  ${torch_include}                        # aten includes
  ${torch_include}/torch/csrc/api/include # torch includes
)

target_link_directories(${exe_name} PUBLIC
  /opt/conda/envs/py_${py_version}/lib/python${py_version}/site-packages/torch/lib # c10, torch
  /opt/rocm/hip/lib
)

target_link_directories(${splitk_exe_name} PUBLIC
  /opt/conda/envs/py_${py_version}/lib/python${py_version}/site-packages/torch/lib # c10, torch
  /opt/rocm/hip/lib
)

target_link_libraries(${exe_name} PUBLIC
  c10
  c10_hip
  torch
  torch_hip
  torch_cpu
  amdhip64
)


target_link_libraries(${splitk_exe_name} PUBLIC
  c10
  c10_hip
  torch
  torch_hip
  torch_cpu
  amdhip64
)

target_compile_definitions(${exe_name} PUBLIC 
  ATTN_FWD_DECODER_MAIN=1
  GLIBCXX_USE_CXX11_ABI=1
  __HIP_PLATFORM_HCC__=1
  USE_ROCM=1
)

target_compile_definitions(${splitk_exe_name} PUBLIC 
  ATTN_FWD_SPLITK_DECODER_MAIN=1
  GLIBCXX_USE_CXX11_ABI=1
  __HIP_PLATFORM_HCC__=1
  USE_ROCM=1
)

include(CMakePrintHelpers)
cmake_print_properties(TARGETS ${exe_name} ${splitk_exe_name} PROPERTIES 
  LINK_LIBRARIES 
  LINK_DIRECTORIES 
  INCLUDE_DIRECTORIES 
  COMPILE_DEFINITIONS 
  COMPILE_OPTIONS
  SOURCES
  HIP_ARCHITECTURES)

rocm_install(TARGETS ${exe_name} ${splitk_exe_name})