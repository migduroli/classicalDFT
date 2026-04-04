include(CTest)
include(Catch)

# Unit tests
file(GLOB_RECURSE UNIT_TEST_SOURCES LIST_DIRECTORIES false unit/*.cpp)

add_executable(classicaldft_unit_tests ${UNIT_TEST_SOURCES})

target_link_libraries(classicaldft_unit_tests
    PRIVATE
        classicaldft
        Catch2::Catch2WithMain
)

target_compile_options(classicaldft_unit_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>
)

target_compile_definitions(classicaldft_unit_tests
    PRIVATE
        DFT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

catch_discover_tests(classicaldft_unit_tests)

# Integration tests
file(GLOB_RECURSE INTEGRATION_TEST_SOURCES LIST_DIRECTORIES false integration/*.cpp)

add_executable(classicaldft_integration_tests ${INTEGRATION_TEST_SOURCES})

target_link_libraries(classicaldft_integration_tests
    PRIVATE
        classicaldft
        Catch2::Catch2WithMain
)

target_compile_options(classicaldft_integration_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>
)

target_include_directories(classicaldft_integration_tests
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

target_compile_definitions(classicaldft_integration_tests
    PRIVATE
        DFT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

catch_discover_tests(classicaldft_integration_tests)
