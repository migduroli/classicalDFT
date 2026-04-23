# Each doc is a standalone executable linked to classicaldft.

macro(add_doc name dir)
    add_executable(doc_${name} ${dir}/main.cpp)
    target_link_libraries(doc_${name} PRIVATE classicaldft)
    target_compile_definitions(doc_${name} PRIVATE
        DOC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/${dir}"
    )
endmacro()

macro(add_check name dir)
    add_executable(check_${name} ${dir}/check/main.cpp)
    target_link_libraries(check_${name} PRIVATE classicaldft)
    target_include_directories(check_${name} PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/${dir}
    )
    target_compile_definitions(check_${name} PRIVATE
        DOC_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/${dir}"
    )
endmacro()

add_doc(console          console)
add_doc(config           config)
add_doc(crystal          crystal)
add_doc(geometry         geometry)
add_doc(arithmetic       math/arithmetic)
add_doc(convolution      math/convolution)
add_doc(fourier          math/fourier)
add_doc(autodiff         math/autodiff)
add_doc(integration      math/integration)
add_doc(spline           math/spline)
add_doc(potentials       potentials)
add_doc(thermodynamics   thermodynamics)
add_doc(fmt              functional/fmt)
add_doc(interaction      functional/interaction)
add_doc(solver           solver)
add_doc(picard           algorithms/picard)
add_doc(string_method    algorithms/string_method)
add_doc(dynamics         dynamics)
add_doc(density          density)
add_doc(nucleation       nucleation)

# Cross-validation checks (compare against Jim's library)
add_check(potentials         potentials)
add_check(thermodynamics     thermodynamics)
add_check(fmt               functional/fmt)
add_check(interaction       functional/interaction)
add_check(solver            solver)
add_check(crystal           crystal)
add_check(density           density)
add_check(dynamics          dynamics)
add_check(nucleation        nucleation)
add_check(string_method     algorithms/string_method)

# Umbrella target to build all cross-validation checks.
add_custom_target(check_all
    DEPENDS
        check_potentials
        check_thermodynamics
        check_fmt
        check_interaction
        check_solver
        check_crystal
        check_density
        check_dynamics
        check_nucleation
        check_string_method
)
