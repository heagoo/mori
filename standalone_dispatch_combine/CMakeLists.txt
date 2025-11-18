cmake_minimum_required(VERSION 3.16)
project(StandaloneDispatchCombine LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(HIP REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${MPI_CXX_INCLUDE_DIRS}
    ${HIP_INCLUDE_DIRS}
)

# Library source files
set(LIB_SOURCES
    src/dispatch_combine.cpp
    src/kernels.hip.cpp
)

# Create library
add_library(dispatch_combine SHARED ${LIB_SOURCES})

# Set HIP compiler for kernel files
set_source_files_properties(src/kernels.hip.cpp PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)

# Link libraries
target_link_libraries(dispatch_combine
    ${MPI_CXX_LIBRARIES}
    hip::host
)

# Set compiler flags
target_compile_options(dispatch_combine PRIVATE
    -Wall -Wextra
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Debug>:-g -O0>
)

# Add HIP architecture flags
set_target_properties(dispatch_combine PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_STANDARD 17
)

# Example executable
add_executable(example examples/example.cpp)
target_link_libraries(example
    dispatch_combine
    ${MPI_CXX_LIBRARIES}
    hip::host
)

# Set RPATH for shared library
set_target_properties(example PROPERTIES
    INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib"
    BUILD_WITH_INSTALL_RPATH FALSE
)

# Install targets
install(TARGETS dispatch_combine example
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

# Print configuration summary
message(STATUS "")
message(STATUS "Configuration Summary:")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "  HIP: ${HIP_VERSION}")
message(STATUS "  MPI: Found")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
