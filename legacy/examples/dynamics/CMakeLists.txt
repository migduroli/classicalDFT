add_executable(example_dynamics main.cpp)
target_link_libraries(example_dynamics PRIVATE classicaldft)
target_compile_definitions(example_dynamics PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
