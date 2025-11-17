cmake_minimum_required(VERSION 3.19)

# Note: This project requires ROCm/HIP and MPI to be installed
# Set ROCm path if not in standard location
if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

project(p2p_allreduce LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(MPI REQUIRED)

# Try to find HIP
find_package(hip QUIET)
if(NOT hip_FOUND)
  message(WARNING "HIP not found. This project requires ROCm/HIP to build.")
  message(WARNING "Please install ROCm from: https://rocm.docs.amd.com/")
  message(WARNING "Set ROCM_PATH to your ROCm installation directory.")
  message(FATAL_ERROR "Cannot proceed without HIP.")
endif()

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
)

# Source files
set(SOURCES
  src/bootstrap.cpp
  src/symmetric_memory.cpp
  src/allreduce.cpp
)

# Create library
add_library(p2p_allreduce ${SOURCES})
target_link_libraries(p2p_allreduce PUBLIC ${MPI_CXX_LIBRARIES} hip::host)
target_compile_options(p2p_allreduce PUBLIC ${MPI_CXX_COMPILE_FLAGS})

# Example executables
add_executable(test_allreduce examples/test_allreduce.cpp)
target_link_libraries(test_allreduce PRIVATE p2p_allreduce)

add_executable(benchmark_allreduce examples/benchmark_allreduce.cpp)
target_link_libraries(benchmark_allreduce PRIVATE p2p_allreduce)

# Installation
install(TARGETS p2p_allreduce test_allreduce benchmark_allreduce
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
