#!/bin/bash
# Build a minimal STATIC OpenCV (core + imgproc only) for the manylinux wheel.
# detector_2d links it directly into the extension .so, so nothing is bundled:
# auditwheel finds no external OpenCV .so to vendor and just applies the
# manylinux tag (+ --strip). Since the package no longer imports cv2, the wheel
# has no opencv-python runtime dependency either.
#
# Linux specifics vs the Windows script:
#   BUILD_SHARED_LIBS=OFF              -> static .a archives
#   CMAKE_POSITION_INDEPENDENT_CODE=ON -> REQUIRED: the static archives (and
#       OpenCV's bundled 3rd-party, e.g. zlib) get linked into a shared .so, so
#       every object must be -fPIC, else the link fails ("recompile with -fPIC").
# Installs to the default prefix (/usr/local); find_package(OpenCV) picks it up
# from the standard search path (no OpenCV_DIR needed). The Linux OpenCVConfig is
# a single config (not the Windows "pack" dispatcher), so static resolves cleanly.
set -eux

# Match the version used on the other platforms.
git clone --depth 1 --branch 4.13.0 https://github.com/opencv/opencv.git

cmake -S opencv -B opencv/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIST=core,imgproc \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
  -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_ITT=OFF \
  -DWITH_OPENCL=OFF -DWITH_FFMPEG=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
  -DWITH_PROTOBUF=OFF -DWITH_QUIRC=OFF -DWITH_ADE=OFF \
  -DWITH_TIFF=OFF -DWITH_JPEG=OFF -DWITH_PNG=OFF -DWITH_WEBP=OFF \
  -DWITH_OPENEXR=OFF -DWITH_JASPER=OFF -DWITH_OPENJPEG=OFF
  # ^ no imgcodecs in BUILD_LIST, so these image-format 3rd-party libs are
  #   unused; OFF avoids OpenCV exporting a static 3rd-party (e.g. liblibtiff.a)
  #   in OpenCVModules.cmake that isn't installed -> "references file ... does
  #   not exist" at the consumer's find_package.

# Cap parallelism (default 2) so the memory-heavy imgproc TUs don't OOM small
# runners; raise CMAKE_BUILD_PARALLEL_LEVEL for more cores/RAM.
cmake --build opencv/build --target install --parallel "${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
