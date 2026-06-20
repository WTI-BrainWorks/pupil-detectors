#!/bin/bash
# Build a minimal OpenCV (core + imgproc only) for the manylinux wheel.
#
# The defaults bloat the bundled lib badly: on x86_64 WITH_IPP=ON statically
# links a large Intel IPPICV blob into libopencv_core, and the libs ship with
# symbols. Disabling IPP/TBB/etc., forcing Release, and stripping (auditwheel
# --strip, see [tool.cibuildwheel.linux].repair-wheel-command) takes the bundled
# OpenCV from ~48MB down to ~10MB.
set -eux

git clone --depth 1 --branch 4.6.0 https://github.com/opencv/opencv.git

cmake -S opencv -B opencv/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_LIST=core,imgproc \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
  -DBUILD_opencv_apps=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
  -DWITH_IPP=OFF -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_ITT=OFF \
  -DWITH_OPENCL=OFF -DWITH_FFMPEG=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
  -DWITH_PROTOBUF=OFF -DWITH_QUIRC=OFF -DWITH_ADE=OFF

cmake --build opencv/build --target install --parallel
