cmake_minimum_required(VERSION 3.20)

project(classicalDFT
    VERSION 2.0.0
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
option(DFT_USE_GRACE      "Enable xmgrace plotting" ON)
option(DFT_USE_MATPLOTLIB "Enable matplotlib-cpp plotting" ON)
option(DFT_CODE_COVERAGE  "Enable code coverage (gcov)" OFF)

# ── Dependencies ─────────────────────────────────────────────────────────────

# GSL
find_package(GSL REQUIRED)

# Armadillo (used in potentials)
find_package(Armadillo REQUIRED)

# FFTW3 (FFT operations)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED IMPORTED_TARGET fftw3)

# nlohmann/json (config parser)
find_package(nlohmann_json 3.11 REQUIRED)

# FetchContent (for GoogleTest, autodiff)
include(FetchContent)

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

# Grace (optional)
if(DFT_USE_GRACE)
    find_library(GRACE_NP_LIB grace_np)
    if(GRACE_NP_LIB)
        message(STATUS "Found grace_np: ${GRACE_NP_LIB}")
    else()
        message(WARNING "grace_np not found; Grace plotting will be unavailable")
        set(DFT_USE_GRACE OFF)
    endif()
endif()

# Google Test (fetched automatically)
if(DFT_BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # Prevent GTest from overriding our compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# matplotlib-cpp (header-only, fetched automatically)
if(DFT_USE_MATPLOTLIB)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    FetchContent_Declare(
        matplotlib_cpp
        GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
        GIT_TAG        master
    )
    # Download source only — skip add_subdirectory to avoid building
    # upstream examples that fail with WITHOUT_NUMPY.
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_GetProperties(matplotlib_cpp)
    if(NOT matplotlib_cpp_POPULATED)
        FetchContent_Populate(matplotlib_cpp)
    endif()

    # Header-only: create interface target manually (avoids building upstream examples)
    add_library(matplotlib_cpp INTERFACE)
    target_include_directories(matplotlib_cpp SYSTEM INTERFACE
        $<BUILD_INTERFACE:${matplotlib_cpp_SOURCE_DIR}>
    )
    target_link_libraries(matplotlib_cpp INTERFACE Python3::Python)

    # Disable numpy C-API to avoid int64_t/long-long template collision on ARM64
    target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()

# ── Core library ─────────────────────────────────────────────────────────────

set(DFT_LIB_SOURCES
    # Math
    src/math/arithmetic.cpp
    src/math/integration.cpp
    src/math/fourier.cpp
    src/math/spline.cpp
    src/math/exceptions.cpp
    # Config
    src/config.cpp
    # Potentials
    src/potentials/potential.cpp
    # Thermodynamics
    src/thermodynamics/eos.cpp
    # Crystal
    src/crystal/lattice.cpp
    # Density & Species
    src/density.cpp
    src/species.cpp
    # Functional
    src/functional/fmt/convolution.cpp
    src/functional/fmt/weights.cpp
    src/functional/fmt/functional.cpp
    src/functional/fmt/species.cpp
    src/functional/interaction.cpp
    # Solver
    src/solver.cpp
    # Dynamics
    src/dynamics/minimizer.cpp
    src/dynamics/fire2.cpp
    src/dynamics/integrator.cpp
    # Plotting
    src/plotting/exceptions.cpp
    # Geometry – base
    src/geometry/base/vertex.cpp
    src/geometry/base/element.cpp
    src/geometry/base/mesh.cpp
    # Geometry – 2D
    src/geometry/2D/element.cpp
    src/geometry/2D/mesh.cpp
    src/geometry/2D/uniform_mesh.cpp
    # Geometry – 3D
    src/geometry/3D/element.cpp
    src/geometry/3D/mesh.cpp
    src/geometry/3D/uniform_mesh.cpp
)

add_library(classicaldft STATIC ${DFT_LIB_SOURCES})

if(DFT_USE_GRACE)
    target_sources(classicaldft PRIVATE src/plotting/grace.cpp)
endif()

target_include_directories(classicaldft
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
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

target_include_directories(classicaldft
    PUBLIC
        ${ARMADILLO_INCLUDE_DIRS}
)

if(DFT_USE_GRACE)
    target_link_libraries(classicaldft PUBLIC ${GRACE_NP_LIB})
    target_compile_definitions(classicaldft PUBLIC DFT_HAS_GRACE)
endif()

if(DFT_USE_MATPLOTLIB)
    target_link_libraries(classicaldft PUBLIC matplotlib_cpp)
    target_compile_definitions(classicaldft PUBLIC DFT_HAS_MATPLOTLIB)
endif()

target_compile_options(classicaldft
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>
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
    add_subdirectory(examples/console)
    add_subdirectory(examples/config)
    add_subdirectory(examples/math/integration)
    add_subdirectory(examples/math/arithmetic)
    add_subdirectory(examples/math/fourier)
    add_subdirectory(examples/math/spline)
    add_subdirectory(examples/geometry)
    add_subdirectory(examples/thermodynamics)
    add_subdirectory(examples/crystal)
    add_subdirectory(examples/density)
    add_subdirectory(examples/species)
    add_subdirectory(examples/functional/fmt)
    add_subdirectory(examples/functional/interaction)
    add_subdirectory(examples/potentials)
    add_subdirectory(examples/solver)
    add_subdirectory(examples/dynamics)
    add_subdirectory(examples/droplet)
    if(DFT_USE_GRACE)
        add_subdirectory(examples/plotting)
    endif()
endif()
