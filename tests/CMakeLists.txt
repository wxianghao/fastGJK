add_executable(testGJK testGJK.cu)
target_include_directories(testGJK PRIVATE ${CMAKE_SOURCE_DIR}/tool-include)
target_link_libraries(testGJK PRIVATE fastGJK::fastGJK)

if(FASTGJK_LARGE_DATASET)
add_executable(testGJK_large testGJK_large.cu)
target_include_directories(testGJK_large PRIVATE ${CMAKE_SOURCE_DIR}/tool-include)
target_link_libraries(testGJK_large PRIVATE fastGJK::fastGJK)
endif()