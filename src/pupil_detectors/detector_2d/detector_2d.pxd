# cython: profile=False, language_level=3
"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
from libcpp.memory cimport shared_ptr

from ..c_types_wrapper cimport Detector2D, Detector2DResult, Mat
from ..detector_base cimport DetectorBase


cdef class Detector2DCore(DetectorBase):
    cdef dict properties
    cdef Detector2D* thisptr
    # prior-seeded ROI tracking state (skip coarse detection while locked on)
    cdef bint _have_prev
    cdef double _prev_cx, _prev_cy, _prev_diam

    cdef shared_ptr[Detector2DResult] c_detect(self, gray_img, color_img=*, roi=*)
    cdef shared_ptr[Detector2DResult] _detect_in_roi(self, roi, Mat frame_image, Mat frameColor, Mat debug_image, should_visualize, color_img, gray_img)
