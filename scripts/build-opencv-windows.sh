#!/usr/bin/env bash
# Build a minimal OpenCV (core + imgproc only) from source so the Windows wheel
# bundles the two small per-module DLLs instead of choco's monolithic
# opencv_world (~68MB, every module). Windows analogue of
# scripts/manylinux-before-all.sh.
#
# Install layout is flattened (bin/, lib/, cmake/) so OpenCV_DIR and the
# delvewheel --add-path don't depend on the MSVC "vc1x" folder name. The
# generator is left unset so CMake uses whichever Visual Studio the runner has
# (same toolset the extension is built with).
set -eux

OPENCV_VERSION=4.13.0
PREFIX="C:/opencv-min"

git clone --depth 1 --branch "$OPENCV_VERSION" https://github.com/opencv/opencv.git

cmake -S opencv -B opencv/build -A x64 \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DBUILD_LIST=core,imgproc \
  -DBUILD_SHARED_LIBS=ON \
  -DOPENCV_BIN_INSTALL_PATH=bin \
  -DOPENCV_LIB_INSTALL_PATH=lib \
  -DOPENCV_3P_LIB_INSTALL_PATH=lib \
  -DOPENCV_CONFIG_INSTALL_PATH=cmake \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
  -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_FFMPEG=OFF \
  -DWITH_PROTOBUF=OFF -DWITH_QUIRC=OFF -DWITH_ADE=OFF

# Cap parallelism (default 2) so memory-heavy TUs don't OOM the runner; raise
# CMAKE_BUILD_PARALLEL_LEVEL for more cores/RAM.
cmake --build opencv/build --config Release --target install --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
