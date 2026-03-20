cmake_minimum_required(VERSION 3.18)
project(fastGJK LANGUAGES CUDA)

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

# Enable to use large dataset
option(FASTGJK_LARGE_DATASET "Use large dataset for testing and benchmarking" OFF)

# Test programs
option(FASTGJK_BUILD_TESTS "Build test programs" ON)
if(FASTGJK_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# # Benchmark programs
# option(FASTGJK_BUILD_BENCHMARKS "Build benchmark programs" ON)
# if(FASTGJK_BUILD_BENCHMARKS)
#     add_subdirectory(benchmarks)
# endif()