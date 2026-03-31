add_executable(example_species main.cpp)
target_link_libraries(example_species PRIVATE classicaldft)
target_compile_definitions(example_species PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
