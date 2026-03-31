add_executable(example_fmt main.cpp)
target_link_libraries(example_fmt PRIVATE classicaldft)
target_compile_definitions(example_fmt PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
