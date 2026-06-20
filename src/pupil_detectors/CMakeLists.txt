cmake_minimum_required(VERSION 3.21...4.0)

project("pupil_detector_base"
    LANGUAGES C CXX
    DESCRIPTION "Pupil Labs Base Pupil Detector"
    HOMEPAGE_URL "https://github.com/pupil-labs/pupil-detectors"
)

# OpenCV 4 requires at least C++11, but we should be fine even selecting C++17 now
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if(APPLE)
    # We target macOS 10.12, which does not offer c++17, but we can use c++1z instead.
    # See https://clang.llvm.org/cxx_status.html
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
endif()

# apply all recommended speed optimization (note -O3 is typically not recommeded
# as it heavily relies on well-written code)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

if(MSVC)
    # for M_PI
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_USE_MATH_DEFINES")

    # TODO: This is a quick and dirty fix for:
    # https://github.com/pupil-labs/pupil/issues/1331 We should investigate this more
    # and fix it correctly at some point.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_EXTENDED_ALIGNED_STORAGE")
endif()

# scikit-build-core provides FindPython; cython-cmake provides UseCython (both
# pulled in via build-system.requires). No NumPy: the Cython sources use typed
# memoryviews (buffer protocol), not the numpy C-API.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.SABIModule)
find_package(Cython MODULE REQUIRED VERSION 3.0)
include(UseCython)

# scikit-build-core sets SKBUILD_SABI_VERSION (from wheel.py-api) only for abi3
# builds (CPython 3.11+ here); on 3.10 it is unset and we build a normal
# version-specific module. USE_SABI then defines Py_LIMITED_API + .abi3 suffix.
# _sabi_args is inherited by the detector_2d subdirectory.
if(DEFINED SKBUILD_SABI_VERSION AND NOT SKBUILD_SABI_VERSION STREQUAL "")
    set(_sabi_args USE_SABI ${SKBUILD_SABI_VERSION})
else()
    set(_sabi_args)
endif()

cython_transpile(detector_base.pyx LANGUAGE CXX CYTHON_ARGS -3
    OUTPUT_VARIABLE detector_base_cxx)
python_add_library(detector_base MODULE WITH_SOABI ${_sabi_args}
    "${detector_base_cxx}")
install(TARGETS detector_base DESTINATION "pupil_detectors")

add_subdirectory(detector_2d)
