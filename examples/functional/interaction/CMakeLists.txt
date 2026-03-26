add_executable(example_interaction main.cpp)
target_link_libraries(example_interaction PRIVATE classicaldft)
target_compile_definitions(example_interaction PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
