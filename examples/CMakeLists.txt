add_executable(bench bench.cu)
target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas -v>)
target_include_directories(bench PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bench PRIVATE fastGJK::fastGJK)

add_executable(test test.cu)
target_compile_options(test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas -v>)
target_include_directories(test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test PRIVATE fastGJK::fastGJK)