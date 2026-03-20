add_executable(testGJK testGJK.cu)
target_include_directories(testGJK PRIVATE ${CMAKE_SOURCE_DIR}/tool-include)
target_link_libraries(testGJK PRIVATE fastGJK::fastGJK)