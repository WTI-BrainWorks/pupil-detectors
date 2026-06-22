cmake_minimum_required(VERSION 3.21...4.0)
cmake_policy(SET CMP0076 NEW)

project("pupil_detector_2d"
    LANGUAGES C CXX
    DESCRIPTION "Pupil Labs 2D Pupil Detector"
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

# Python, Cython and UseCython are established by the parent CMakeLists before
# add_subdirectory(); re-running find_package(Cython) here would fail trying to
# redefine the Cython::Cython imported target. Only the deps unique to this
# module are resolved here.
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED core imgproc) # add highgui to use imshow/waitKey

# detector_2d.pyx cimports across the package (..c_types_wrapper, ..coarse_pupil,
# ..detector_base); -I on the package root lets Cython resolve them.
cython_transpile(detector_2d.pyx LANGUAGE CXX
    CYTHON_ARGS -3 -I "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE detector_2d_cxx)
python_add_library(detector_2d MODULE WITH_SOABI ${_sabi_args}
    "${detector_2d_cxx}")

target_include_directories(
    detector_2d
    PUBLIC

    # external
    ${OpenCV_INCLUDE_DIRS}

    # local
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cpp/shared/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cpp/singleeyefitter>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cpp/singleeyefitter/ImageProcessing>
)

target_sources(
    detector_2d
    PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cpp/singleeyefitter/detectorUtils.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cpp/singleeyefitter/utils.cpp>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../cpp/singleeyefitter/ImageProcessing/cvx.cpp>
)

target_link_libraries(detector_2d PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)

# Function-level dead-code elimination when statically linking OpenCV (Linux).
# GNU ld otherwise pulls whole object files from the static archives (it has no
# per-function stripping like MSVC's default /OPT:REF), roughly doubling the
# .so. Pairing per-function/data sections with --gc-sections drops the
# unreferenced OpenCV code. Linux-only: MSVC already strips, and macOS links a
# dynamic OpenCV with ld64 (which uses -dead_strip, not --gc-sections).
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_compile_options(detector_2d PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(detector_2d PRIVATE -Wl,--gc-sections)
endif()

install(TARGETS detector_2d DESTINATION "pupil_detectors/detector_2d")
