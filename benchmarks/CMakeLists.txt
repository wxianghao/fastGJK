include(FetchContent)

FetchContent_Declare(
  nvbench
  GIT_REPOSITORY https://github.com/NVIDIA/nvbench.git
  GIT_TAG        python-0.2.0
)
FetchContent_MakeAvailable(nvbench)

add_executable(benchGJK benchGJK.cu)
target_include_directories(benchGJK PRIVATE ${CMAKE_SOURCE_DIR}/tool-include)
target_link_libraries(benchGJK PRIVATE fastGJK::fastGJK nvbench::main)
if(FASTGJK_LARGE_DATASET)
target_compile_definitions(benchGJK PRIVATE FASTGJK_LARGE_DATASET)
endif()