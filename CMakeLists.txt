cmake_minimum_required(VERSION 3.20)

project(classicalDFT
    VERSION 3.0.0
    DESCRIPTION "Classical Density Functional Theory library"
    LANGUAGES CXX
)

# ── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──────────────────────────────────────────────────────────────────
option(DFT_BUILD_TESTS    "Build unit tests"       ON)
option(DFT_BUILD_EXAMPLES "Build example programs"  ON)
option(DFT_CODE_COVERAGE  "Enable code coverage (gcov)" OFF)

# ── Dependencies ─────────────────────────────────────────────────────────────

# GSL
find_package(GSL REQUIRED)

# Armadillo
find_package(Armadillo REQUIRED)

# FFTW3
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

# nlohmann/json
find_package(nlohmann_json 3.11 REQUIRED)

# FetchContent
include(FetchContent)

# autodiff (header-only, forward-mode AD)
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

# Google Test
if(DFT_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ── Core library ─────────────────────────────────────────────────────────────

set(DFT_LIB_SOURCES
    # Config / IO
    src/cdft/config.cpp
    # Numerics
    src/cdft/numerics/fourier.cpp
    src/cdft/numerics/math.cpp
    src/cdft/numerics/spline.cpp
    # Physics
    src/cdft/physics/potentials.cpp
    src/cdft/physics/eos.cpp
    src/cdft/physics/crystal.cpp
    # Functional
    src/cdft/functional/density.cpp
    src/cdft/functional/fmt.cpp
    src/cdft/functional/species.cpp
    src/cdft/functional/interaction.cpp
)

add_library(classicaldft STATIC ${DFT_LIB_SOURCES})

target_include_directories(classicaldft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${ARMADILLO_INCLUDE_DIRS}
)

target_link_libraries(classicaldft
    PUBLIC
        GSL::gsl
        GSL::gslcblas
        ${ARMADILLO_LIBRARIES}
        PkgConfig::FFTW3
        nlohmann_json::nlohmann_json
        autodiff::autodiff
)

target_compile_options(classicaldft
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
)

# ── Coverage ─────────────────────────────────────────────────────────────────
if(DFT_CODE_COVERAGE)
    target_compile_options(classicaldft PUBLIC --coverage -fprofile-arcs -ftest-coverage)
    target_link_options(classicaldft PUBLIC --coverage)
endif()

# ── Tests ────────────────────────────────────────────────────────────────────
if(DFT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ── Examples ─────────────────────────────────────────────────────────────────
if(DFT_BUILD_EXAMPLES)
    add_subdirectory(examples/cdft)
endif()
