cmake_minimum_required(VERSION 3.20)

project(classicalDFT
    VERSION 3.0.0
    DESCRIPTION "Classical Density Functional Theory library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(DFT_BUILD_TESTS    "Build unit tests"        ON)
option(DFT_BUILD_EXAMPLES "Build example programs"   OFF)
option(DFT_USE_GRACE      "Enable xmgrace plotting"  OFF)
option(DFT_USE_MATPLOTLIB "Enable matplotlib-cpp"     OFF)
option(DFT_CODE_COVERAGE  "Enable code coverage"      OFF)

# Dependencies
find_package(Armadillo REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
pkg_check_modules(GSL REQUIRED gsl)

include(FetchContent)

# nlohmann/json
find_package(nlohmann_json 3.11.0 REQUIRED)

# autodiff (header-only forward-mode automatic differentiation)
FetchContent_Declare(
    autodiff
    GIT_REPOSITORY https://github.com/autodiff/autodiff.git
    GIT_TAG        v1.1.2
)
set(AUTODIFF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(AUTODIFF_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(autodiff)

# Catch2 v3
if(DFT_BUILD_TESTS)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
    )
    FetchContent_MakeAvailable(Catch2)
endif()

# Core library (static, grows as phases land)
set(DFT_LIB_SOURCES
    src/crystal.cpp
    src/config/parser.cpp
    src/math/fourier.cpp
    src/math/spline.cpp
    src/physics/eos.cpp
)

add_library(classicaldft STATIC ${DFT_LIB_SOURCES})

target_include_directories(classicaldft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${FFTW3_INCLUDE_DIRS}
        ${GSL_INCLUDE_DIRS}
)

target_link_directories(classicaldft
    PUBLIC
        ${FFTW3_LIBRARY_DIRS}
        ${GSL_LIBRARY_DIRS}
)

target_link_libraries(classicaldft
    PUBLIC
        ${ARMADILLO_LIBRARIES}
        nlohmann_json::nlohmann_json
        ${FFTW3_LIBRARIES}
        ${GSL_LIBRARIES}
        autodiff::autodiff
)

target_include_directories(classicaldft
    PUBLIC
        ${ARMADILLO_INCLUDE_DIRS}
)

target_compile_options(classicaldft
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>
)

# Coverage
if(DFT_CODE_COVERAGE)
    target_compile_options(classicaldft PUBLIC --coverage -fprofile-arcs -ftest-coverage)
    target_link_options(classicaldft PUBLIC --coverage)
endif()

# Tests
if(DFT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
