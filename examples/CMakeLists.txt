add_executable(bench bench.cu)
target_compile_options(bench PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xptxas -v>)
target_link_libraries(bench PRIVATE fastGJK::fastGJK)