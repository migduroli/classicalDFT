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
option(DFT_CODE_COVERAGE  "Enable code coverage (gcov)" OFF)

# ── Dependencies ─────────────────────────────────────────────────────────────

# GSL
find_package(GSL REQUIRED)

# Boost (property_tree for config_parser, serialization for potentials)
find_package(Boost REQUIRED COMPONENTS serialization)

# Armadillo (used in potentials)
find_package(Armadillo REQUIRED)

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
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    # Prevent GTest from overriding our compiler/linker settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

# ── Core library ─────────────────────────────────────────────────────────────

set(DFT_LIB_SOURCES
    # Exceptions
    src/exceptions/grace.cpp
    src/exceptions/parameter.cpp
    # IO
    src/io/config.cpp
    # Numerics
    src/numerics/arithmetic.cpp
    src/numerics/integration.cpp
    # Physics
    src/physics/potentials/intermolecular/potential.cpp
    # Geometry – base
    src/geometry/base/vertex.cpp
    src/geometry/base/element.cpp
    src/geometry/base/mesh.cpp
    # Geometry – 2D
    src/geometry/2D/element.cpp
    src/geometry/2D/mesh.cpp
    # Geometry – 3D
    src/geometry/3D/element.cpp
    src/geometry/3D/mesh.cpp
)

add_library(classicaldft STATIC ${DFT_LIB_SOURCES})

if(DFT_USE_GRACE)
    target_sources(classicaldft PRIVATE src/graph/grace.cpp)
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
        Boost::serialization
        ${ARMADILLO_LIBRARIES}
)

target_include_directories(classicaldft
    PUBLIC
        ${ARMADILLO_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
)

if(DFT_USE_GRACE)
    target_link_libraries(classicaldft PUBLIC ${GRACE_NP_LIB})
    target_compile_definitions(classicaldft PUBLIC DFT_HAS_GRACE)
endif()

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
    add_subdirectory(examples/console)
    add_subdirectory(examples/config)
    add_subdirectory(examples/numerics/integration)
    add_subdirectory(examples/numerics/arithmetic)
    add_subdirectory(examples/geometry)
    if(DFT_USE_GRACE)
        add_subdirectory(examples/graphs)
        add_subdirectory(examples/physics/potentials/intermolecular)
    endif()
endif()
