#!/usr/bin/env bash
# Build a minimal STATIC OpenCV (core + imgproc only) from source. detector_2d
# links it directly into the .pyd (find_package(OpenCV core imgproc) +
# ${OpenCV_LIBS}), so the Windows wheel is fully self-contained: no bundled
# opencv DLLs, no delvewheel, no add_dll_directory, and -- since the package no
# longer imports cv2 -- no opencv-python runtime dependency. With /OPT:REF dead-
# code elimination only the used OpenCV code lands in the ~5MB .pyd (~1.7MB
# wheel). Windows analogue of scripts/manylinux-before-all.sh.
#
# Two flags carry the load:
#   BUILD_SHARED_LIBS=OFF      -> static .lib archives instead of DLLs
#   BUILD_WITH_STATIC_CRT=OFF  -> /MD dynamic CRT, matching the Python extension
#                                 (mixing /MT here would clash with Python's /MD)
# Use the DEFAULT install layout (x64/vc<NN>/staticlib) and point OpenCV_DIR
# (set in pyproject) straight at that leaf static config. Do NOT flatten the
# install: the flattened top-level OpenCVConfig.cmake is a "Windows pack"
# dispatcher that auto-detects shared libs and fails for a static build
# ("no binaries compatible with your configuration"). The leaf config knows it
# is static and links correctly. Generator left unset so CMake uses whichever
# Visual Studio the runner has (vc17 on windows-latest / VS2022).
set -eux

OPENCV_VERSION=4.13.0
PREFIX="C:/opencv-min"

git clone --depth 1 --branch "$OPENCV_VERSION" https://github.com/opencv/opencv.git

cmake -S opencv -B opencv/build -A x64 \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_LIST=core,imgproc \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_WITH_STATIC_CRT=OFF \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
  -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_FFMPEG=OFF \
  -DWITH_GSTREAMER=OFF -DWITH_PROTOBUF=OFF -DWITH_QUIRC=OFF -DWITH_ADE=OFF \
  -DWITH_EIGEN=OFF -DWITH_LAPACK=OFF -DWITH_1394=OFF -DWITH_DSHOW=OFF \
  -DWITH_OPENEXR=OFF -DWITH_TIFF=OFF -DWITH_JPEG=OFF -DWITH_PNG=OFF \
  -DWITH_WEBP=OFF -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF \
  -DWITH_ITT=OFF -DWITH_OPENCL=OFF
  # ^ ITT/OpenCL are unused here; off => fewer statically-linked 3rd-party blobs
  #   (leaves just OpenCV[Apache-2.0] + zlib[zlib license] to attribute).

# Cap parallelism (default 2) so memory-heavy TUs don't OOM the runner; raise
# CMAKE_BUILD_PARALLEL_LEVEL for more cores/RAM.
cmake --build opencv/build --config Release --target install --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
