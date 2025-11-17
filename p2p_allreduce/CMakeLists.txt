cmake_minimum_required(VERSION 3.19)

project(p2p_allreduce LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find dependencies
find_package(MPI REQUIRED)
find_package(hip REQUIRED)

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

# Example executable
add_executable(test_allreduce examples/test_allreduce.cpp)
target_link_libraries(test_allreduce PRIVATE p2p_allreduce)

# Installation
install(TARGETS p2p_allreduce test_allreduce
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
