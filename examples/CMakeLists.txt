# Each example is a standalone executable linked to classicaldft.

macro(add_example name dir)
    add_executable(example_${name} ${dir}/main.cpp)
    target_link_libraries(example_${name} PRIVATE classicaldft)
    target_compile_definitions(example_${name} PRIVATE
        EXAMPLE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/${dir}"
    )
endmacro()

add_example(console          console)
add_example(config           config)
add_example(crystal          crystal)
add_example(geometry         geometry)
add_example(arithmetic       math/arithmetic)
add_example(fourier          math/fourier)
add_example(integration      math/integration)
add_example(spline           math/spline)
add_example(potentials       potentials)
add_example(thermodynamics   thermodynamics)
add_example(fmt              functional/fmt)
add_example(interaction      functional/interaction)
add_example(solver           solver)
add_example(dynamics         dynamics)
add_example(density          density)
add_example(nucleation       nucleation)
