add_executable(example_crystal main.cpp)
target_link_libraries(example_crystal PRIVATE classicaldft)
target_compile_definitions(example_crystal PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
