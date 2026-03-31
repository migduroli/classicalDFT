add_executable(example_solver main.cpp)
target_link_libraries(example_solver PRIVATE classicaldft)
target_compile_definitions(example_solver PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
