cmake_minimum_required(VERSION 3.19)

# Note: This project requires ROCm/HIP and MPI to be installed
# Set ROCm path if not in standard location
if(NOT DEFINED ROCM_PATH)
  set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation")
endif()

list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH})

# Enable HIP language for GPU kernel compilation
project(p2p_allreduce LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set HIP standard
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# Find dependencies
find_package(MPI REQUIRED)

# Try to find HIP
find_package(hip REQUIRED)

# Include directories
include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MPI_CXX_INCLUDE_DIRS}
)

# Source files - mark as HIP sources to compile with hipcc
set(HIP_SOURCES
  src/bootstrap.cpp
  src/symmetric_memory.cpp
  src/allreduce.cpp
)

# Set the language for HIP source files
set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

# Create library with HIP support
add_library(p2p_allreduce ${HIP_SOURCES})
target_link_libraries(p2p_allreduce PUBLIC ${MPI_CXX_LIBRARIES} hip::device hip::host)
target_compile_options(p2p_allreduce PUBLIC ${MPI_CXX_COMPILE_FLAGS})

# Example executables - also need HIP compilation
set_source_files_properties(examples/test_allreduce.cpp PROPERTIES LANGUAGE HIP)
add_executable(test_allreduce examples/test_allreduce.cpp)
target_link_libraries(test_allreduce PRIVATE p2p_allreduce hip::device)

set_source_files_properties(examples/benchmark_allreduce.cpp PROPERTIES LANGUAGE HIP)
add_executable(benchmark_allreduce examples/benchmark_allreduce.cpp)
target_link_libraries(benchmark_allreduce PRIVATE p2p_allreduce hip::device)

# Installation
install(TARGETS p2p_allreduce test_allreduce benchmark_allreduce
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
