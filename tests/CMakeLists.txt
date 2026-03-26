file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.cpp)

add_executable(classicaldft_tests ${TEST_SOURCES})

target_link_libraries(classicaldft_tests
    PRIVATE
        classicaldft
        GTest::gtest
)

add_test(NAME classicaldft_tests COMMAND classicaldft_tests)

# Copy config files needed by config_parser tests
configure_file(io/config_files/config.ini  ${CMAKE_CURRENT_BINARY_DIR}/config.ini  COPYONLY)
configure_file(io/config_files/config.json ${CMAKE_CURRENT_BINARY_DIR}/config.json COPYONLY)
configure_file(io/config_files/config.xml  ${CMAKE_CURRENT_BINARY_DIR}/config.xml  COPYONLY)
configure_file(io/config_files/config.info ${CMAKE_CURRENT_BINARY_DIR}/config.info COPYONLY)