cmake_minimum_required(VERSION 3.18)
project(fastGJK LANGUAGES CXX CUDA)

# Set language standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Header-only interface library
add_library(fastGJK INTERFACE)
target_include_directories(fastGJK INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_library(fastGJK::fastGJK ALIAS fastGJK)

# Option for double precision
option(FASTGJK_USE_DOUBLE "Use double precision floating point" OFF)
if(FASTGJK_USE_DOUBLE)
    target_compile_definitions(fastGJK INTERFACE USE_DOUBLE)
endif()

# Require CUDA for consumers
target_compile_features(fastGJK INTERFACE cuda_std_14)

# Test programs
option(FASTGJK_BUILD_TESTS "Build test programs" ON)
if(FASTGJK_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# Third-party: openGJK (scalar + GPU)
set(BUILD_SCALAR ON CACHE BOOL "" FORCE)
set(BUILD_SIMD OFF CACHE BOOL "" FORCE)
set(BUILD_GPU ON CACHE BOOL "" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(USE_32BITS ON CACHE BOOL "" FORCE)
add_subdirectory(third-parties/openGJK)

# Benchmark programs
option(FASTGJK_BUILD_BENCHMARKS "Build benchmark programs" ON)
if(FASTGJK_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()