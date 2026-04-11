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
option(DFT_USE_OPENMP     "Enable OpenMP parallel loops" ON)
option(DFT_CODE_COVERAGE  "Enable code coverage"      OFF)

# Dependencies
find_package(Armadillo REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3 REQUIRED fftw3)
pkg_check_modules(GSL REQUIRED gsl)

if(DFT_USE_OPENMP)
    find_package(OpenMP QUIET)

    if(NOT OpenMP_CXX_FOUND AND APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang")
        find_program(HOMEBREW_BREW brew)
        set(_libomp_prefix "")

        if(HOMEBREW_BREW)
            execute_process(
                COMMAND ${HOMEBREW_BREW} --prefix libomp
                OUTPUT_VARIABLE _libomp_prefix
                OUTPUT_STRIP_TRAILING_WHITESPACE
                RESULT_VARIABLE _libomp_result
                ERROR_QUIET
            )
            if(NOT _libomp_result EQUAL 0)
                set(_libomp_prefix "")
            endif()
        endif()

        if(NOT _libomp_prefix)
            foreach(_candidate /opt/homebrew/opt/libomp /usr/local/opt/libomp)
                if(EXISTS "${_candidate}/include/omp.h" AND EXISTS "${_candidate}/lib/libomp.dylib")
                    set(_libomp_prefix "${_candidate}")
                    break()
                endif()
            endforeach()
        endif()

        if(_libomp_prefix)
            add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
            target_compile_options(OpenMP::OpenMP_CXX INTERFACE -Xclang -fopenmp)
            target_link_options(OpenMP::OpenMP_CXX INTERFACE -Xclang -fopenmp)
            target_include_directories(OpenMP::OpenMP_CXX SYSTEM INTERFACE "${_libomp_prefix}/include")
            target_link_libraries(OpenMP::OpenMP_CXX INTERFACE "${_libomp_prefix}/lib/libomp.dylib")
            set(OpenMP_CXX_FOUND TRUE)
            message(STATUS "OpenMP: using Homebrew libomp from ${_libomp_prefix}")
        endif()
    endif()
endif()

# FFTW3 threads (optional, enables multi-threaded FFTs)
find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads HINTS ${FFTW3_LIBRARY_DIRS})

include(FetchContent)

# nlohmann/json
find_package(nlohmann_json 3.11.0 REQUIRED)

# toml++ (header-only TOML parser)
FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(tomlplusplus)

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
        tomlplusplus::tomlplusplus
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

if(DFT_USE_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(classicaldft PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP: enabled")
elseif(DFT_USE_OPENMP)
    message(STATUS "OpenMP: not found, building without loop parallelism")
else()
    message(STATUS "OpenMP: disabled")
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
