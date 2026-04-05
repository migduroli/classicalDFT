cmake_minimum_required(VERSION 3.20)

project(libclassicaldft
    VERSION 3.0.0
    DESCRIPTION "Modern classicalDFT — a C++23 library for classical density functional theory"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(DFT_BUILD_TESTS    "Build unit tests"        ON)
option(DFT_BUILD_DOCS    "Build doc programs"        OFF)
option(DFT_USE_GRACE      "Enable xmgrace plotting"  OFF)
option(DFT_USE_MATPLOTLIB "Enable matplotlib-cpp"     OFF)
option(DFT_CODE_COVERAGE  "Enable code coverage"      OFF)

# Dependencies
find_package(Armadillo REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
pkg_check_modules(GSL REQUIRED gsl)

# FFTW3 threads (optional, enables multi-threaded FFTs)
find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads HINTS ${FFTW3_LIBRARY_DIRS})

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

# Suppress warnings from autodiff headers (third-party code).
get_target_property(_autodiff_dirs autodiff INTERFACE_INCLUDE_DIRECTORIES)
set_target_properties(autodiff PROPERTIES INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_autodiff_dirs}")

# matplotlib-cpp (header-only, fetched automatically)
if(DFT_USE_MATPLOTLIB)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    FetchContent_Declare(
        matplotlib_cpp
        GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
        GIT_TAG        master
    )
    if(POLICY CMP0169)
        cmake_policy(SET CMP0169 OLD)
    endif()
    FetchContent_GetProperties(matplotlib_cpp)
    if(NOT matplotlib_cpp_POPULATED)
        FetchContent_Populate(matplotlib_cpp)
    endif()

    add_library(matplotlib_cpp INTERFACE)
    target_include_directories(matplotlib_cpp SYSTEM INTERFACE
        $<BUILD_INTERFACE:${matplotlib_cpp_SOURCE_DIR}>
    )
    target_link_libraries(matplotlib_cpp INTERFACE Python3::Python)
    target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()

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

# FFTW3 multi-threaded FFTs
if(FFTW3_THREADS_LIBRARY)
    target_link_libraries(classicaldft PUBLIC ${FFTW3_THREADS_LIBRARY})
    target_compile_definitions(classicaldft PUBLIC DFT_FFTW_THREADS)
    message(STATUS "FFTW3 threads: enabled (${FFTW3_THREADS_LIBRARY})")
else()
    message(STATUS "FFTW3 threads: not found, single-threaded FFTs")
endif()

target_include_directories(classicaldft
    PUBLIC
        ${ARMADILLO_INCLUDE_DIRS}
)

target_compile_options(classicaldft
    PRIVATE
        $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
        $<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>
)

# Matplotlib
if(DFT_USE_MATPLOTLIB)
    target_link_libraries(classicaldft PUBLIC matplotlib_cpp)
    target_compile_definitions(classicaldft PUBLIC DFT_HAS_MATPLOTLIB)
endif()

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

# Docs
if(DFT_BUILD_DOCS)
    add_subdirectory(docs)
endif()
