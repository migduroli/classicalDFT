add_executable(example_density main.cpp)
target_link_libraries(example_density PRIVATE classicaldft)
target_compile_definitions(example_density PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
