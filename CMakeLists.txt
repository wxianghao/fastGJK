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

# Example programs
option(FASTGJK_BUILD_EXAMPLES "Build example programs" ON)
if(FASTGJK_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()