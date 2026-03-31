add_executable(example_droplet main.cpp)
target_link_libraries(example_droplet PRIVATE classicaldft)
target_compile_definitions(example_droplet PRIVATE
    EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
