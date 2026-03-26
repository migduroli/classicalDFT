add_executable(example_potentials main.cpp)
target_link_libraries(example_potentials PRIVATE classicaldft)
target_compile_definitions(example_potentials PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
