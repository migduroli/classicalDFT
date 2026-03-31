add_executable(example_geometry main.cpp)
target_link_libraries(example_geometry PRIVATE classicaldft)
target_compile_definitions(example_geometry PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
