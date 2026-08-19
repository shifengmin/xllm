# Host-only: cmake -S . -B build && cmake --build build && ./build/sparse_broadcast_layout_test
# NPU:       ASCEND_HOME_PATH=... SHMEM_ROOT=... cmake -S . -B build && cmake --build build && ./run.sh

cmake_minimum_required(VERSION 3.16)

if(DEFINED ENV{ASCEND_HOME_PATH})
  set(CMAKE_C_COMPILER bisheng)
  set(CMAKE_CXX_COMPILER bisheng)
endif()

project(sparse_broadcast LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(SPARSE_BCAST_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(sparse_broadcast_layout_test sparse_broadcast_layout_test.cpp)
target_include_directories(sparse_broadcast_layout_test PRIVATE ${SPARSE_BCAST_DIR})
if(DEFINED ENV{ASCEND_HOME_PATH})
  target_compile_options(sparse_broadcast_layout_test PRIVATE -xc++)
endif()

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
  message(STATUS "ASCEND_HOME_PATH unset: building host-only layout test")
  return()
endif()

if(NOT DEFINED SHMEM_ROOT)
  if(DEFINED ENV{SHMEM_ROOT})
    set(SHMEM_ROOT $ENV{SHMEM_ROOT})
  elseif(DEFINED ENV{ACLSHMEM_HOME})
    set(SHMEM_ROOT $ENV{ACLSHMEM_HOME})
  else()
    message(FATAL_ERROR "Set SHMEM_ROOT or ACLSHMEM_HOME to the CANN SHMEM source or install prefix")
  endif()
endif()

set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
if(NOT DEFINED SOC_TYPE)
  set(SOC_TYPE "Ascend910B")
endif()

set(TARGET_TIMESTAMP 20260327)
set(CURRENT_TIMESTAMP 0)
set(IS_NEW_VERSION FALSE)
execute_process(
  COMMAND bisheng -v
  OUTPUT_VARIABLE BISHENG_STDOUT
  ERROR_VARIABLE BISHENG_STDERR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)
set(BISHENG_VERSION_OUTPUT "${BISHENG_STDOUT} ${BISHENG_STDERR}")
if(BISHENG_VERSION_OUTPUT)
  string(REGEX MATCH "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])" DATE_MATCH "${BISHENG_VERSION_OUTPUT}")
  if(DATE_MATCH)
    string(REGEX REPLACE "([0-9][0-9][0-9][0-9])-([0-9][0-9])-([0-9][0-9])" "\\1\\2\\3" CURRENT_TIMESTAMP "${DATE_MATCH}")
    if(CURRENT_TIMESTAMP GREATER_EQUAL 20260327)
      set(IS_NEW_VERSION TRUE)
    endif()
  endif()
endif()

if(IS_NEW_VERSION)
  if(SOC_TYPE STREQUAL "Ascend950")
    set(NPU_ARCH dav-3510)
    set(FFTS_COMPILE_FLAG "")
  else()
    set(NPU_ARCH dav-2201)
    set(FFTS_COMPILE_FLAG --cce-auto-set-ffts=false)
    add_compile_definitions(FFTS_CONFIG=1)
  endif()
  set(CCE_AICORE_ARCH_VEC "")
  set(CMAKE_CCE_COMPILE_OPTIONS
    -xasc
    -Xhost-start -ftrapv -Xhost-end
    --npu-arch=${NPU_ARCH}
    --cce-auto-infer-kernel-type=false
    ${FFTS_COMPILE_FLAG}
  )
else()
  if(SOC_TYPE STREQUAL "Ascend950")
    set(CCE_AICORE_ARCH_VEC --cce-aicore-arch=dav-c310-vec)
  else()
    set(CCE_AICORE_ARCH_VEC --cce-aicore-arch=dav-c220-vec)
    add_compile_definitions(FFTS_CONFIG=1)
  endif()
  set(CMAKE_CCE_COMPILE_OPTIONS
    -xcce
    -Xhost-start -ftrapv -Xhost-end
    "SHELL:-mllvm -cce-aicore-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-function-stack-size=0x8000"
    "SHELL:-mllvm -cce-aicore-record-overflow=true"
    "SHELL:-mllvm -cce-aicore-addr-transform"
    "SHELL:-mllvm -cce-aicore-dcci-insert-for-scalar=false"
  )
endif()

set(CMAKE_CPP_COMPILE_OPTIONS
  -xc++
  "SHELL:-include stdint.h"
  "SHELL:-include stddef.h"
)

include_directories(
  ${SPARSE_BCAST_DIR}
  ${SHMEM_ROOT}/include
  ${SHMEM_ROOT}/src
  ${SHMEM_ROOT}/src/device
  ${SHMEM_ROOT}/src/device_simt
  ${ASCEND_HOME_PATH}/compiler/tikcpp
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl
  ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface
  ${ASCEND_HOME_PATH}/include
  ${ASCEND_HOME_PATH}/include/experiment/runtime
  ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/include
  ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp
  ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw
  ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/impl
  ${ASCEND_HOME_PATH}/${CMAKE_SYSTEM_PROCESSOR}-linux/tikcpp/tikcfw/interface
)

link_directories(${ASCEND_HOME_PATH}/lib64)

set(SHMEM_LIB_DIR "")
foreach(CANDIDATE
    ${SHMEM_ROOT}/build/lib
    ${SHMEM_ROOT}/lib
    ${SHMEM_ROOT}/install/shmem/lib)
  if(EXISTS ${CANDIDATE}/libshmem.so)
    set(SHMEM_LIB_DIR ${CANDIDATE})
    break()
  endif()
endforeach()
if(SHMEM_LIB_DIR STREQUAL "")
  message(FATAL_ERROR "libshmem.so not found under ${SHMEM_ROOT}. Build SHMEM first or set SHMEM_ROOT to an install prefix.")
endif()
link_directories(${SHMEM_LIB_DIR})
message(STATUS "SHMEM_ROOT=${SHMEM_ROOT}")
message(STATUS "SHMEM_LIB_DIR=${SHMEM_LIB_DIR}")

add_library(sparse_broadcast_kernel SHARED sparse_broadcast_kernel.cpp)
target_compile_options(sparse_broadcast_kernel PRIVATE
  ${CMAKE_CCE_COMPILE_OPTIONS}
  ${CCE_AICORE_ARCH_VEC}
  -O2
  -std=c++17
  -DL2_CACHE_HINT
)
target_link_options(sparse_broadcast_kernel PRIVATE --cce-fatobj-link ${CCE_AICORE_ARCH_VEC})
target_link_libraries(sparse_broadcast_kernel PRIVATE shmem runtime ascendcl c_sec pthread dl)

add_executable(sparse_broadcast_test sparse_broadcast_test.cpp)
target_compile_options(sparse_broadcast_test PRIVATE ${CMAKE_CPP_COMPILE_OPTIONS} -O2 -std=c++17)
target_link_libraries(sparse_broadcast_test PRIVATE sparse_broadcast_kernel shmem runtime ascendcl c_sec pthread dl)
