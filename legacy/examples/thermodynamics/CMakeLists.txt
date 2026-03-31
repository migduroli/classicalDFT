add_executable(example_thermodynamics main.cpp)
target_link_libraries(example_thermodynamics PRIVATE classicaldft)
target_compile_definitions(example_thermodynamics PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
