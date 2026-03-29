set(TEST_SOURCES
    main.cpp
    cdft/physics/potentials_test.cpp
    cdft/physics/eos_test.cpp
    cdft/physics/crystal_test.cpp
    cdft/numerics/fourier_test.cpp
    cdft/numerics/math_test.cpp
    cdft/numerics/spline_test.cpp
    cdft/functional/fmt_test.cpp
    cdft/functional/density_test.cpp
    cdft/config_test.cpp
)

add_executable(classicaldft_tests ${TEST_SOURCES})

target_link_libraries(classicaldft_tests
    PRIVATE
        classicaldft
        GTest::gtest
)

add_test(NAME classicaldft_tests COMMAND classicaldft_tests)
set_tests_properties(classicaldft_tests PROPERTIES
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Copy config files needed by config_parser tests
configure_file(io/config_files/config.ini  ${CMAKE_CURRENT_BINARY_DIR}/config.ini  COPYONLY)
configure_file(io/config_files/config.json ${CMAKE_CURRENT_BINARY_DIR}/config.json COPYONLY)
configure_file(io/config_files/config.xml  ${CMAKE_CURRENT_BINARY_DIR}/config.xml  COPYONLY)
configure_file(io/config_files/config.info ${CMAKE_CURRENT_BINARY_DIR}/config.info COPYONLY)