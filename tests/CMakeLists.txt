file(GLOB_RECURSE TEST_SOURCES LIST_DIRECTORIES false *.cpp)

add_executable(classicaldft_tests ${TEST_SOURCES})

target_link_libraries(classicaldft_tests
    PRIVATE
        classicaldft
        Catch2::Catch2WithMain
)

target_compile_options(classicaldft_tests
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>
)

target_compile_definitions(classicaldft_tests
    PRIVATE
        DFT_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

include(CTest)
include(Catch)
catch_discover_tests(classicaldft_tests)
