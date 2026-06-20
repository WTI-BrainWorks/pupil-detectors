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
import typing as T

import cv2
import numpy as np

from cython.operator cimport dereference as deref
from libc.math cimport M_PI
from libcpp.memory cimport shared_ptr

from ..c_types_wrapper cimport (
    CV_8UC1,
    CV_8UC3,
    Detector2D,
    Detector2DResult,
    Mat,
    Rect_,
)
from ..coarse_pupil cimport center_surround
from ..detector_base cimport DetectorBase

from ..roi import Roi


cdef class Detector2DCore(DetectorBase):

    def __cinit__(self, *args, **kwargs):
        self.thisptr = new Detector2D()
        self._have_prev = False

    def __dealloc__(self):
        del self.thisptr

    def __init__(self, properties = None):
        # initialize with defaults first and then update
        self.properties = self.get_default_properties()
        if properties is not None:
            self.update_properties(properties)

    @staticmethod
    def get_default_properties():
        return {
            "coarse_detection": True,
            "coarse_filter_min": 128,
            "coarse_filter_max": 280,
            "intensity_range": 23,
            "blur_size": 5,
            "canny_treshold": 160,
            "canny_ration": 2,
            "canny_aperture": 5,
            "pupil_size_max": 100,
            "pupil_size_min": 10,
            "strong_perimeter_ratio_range_min": 0.8,
            "strong_perimeter_ratio_range_max": 1.1,
            "strong_area_ratio_range_min": 0.6,
            "strong_area_ratio_range_max": 1.1,
            "contour_size_min": 5,
            "ellipse_roundness_ratio": 0.1,
            "initial_ellipse_fit_treshhold": 1.8,
            "final_perimeter_ratio_range_min": 0.6,
            "final_perimeter_ratio_range_max": 1.2,
            "ellipse_true_support_min_dist": 2.5,
            "support_pixel_ratio_exponent": 2.0,
            # Max evaluations in the combinatorial contour-combination search.
            # Bounds the latency tail on fragmented (e.g. blink) frames; the
            # original value was 1000. See detect_2d.hpp.
            "combine_evals_max": 100,
            # Structural fallback: when the edge/contour path fails, fit the
            # ellipse to the largest dark blob (robust to glint-fragmented
            # boundaries). 0 disables.
            "use_blob_fallback": 1,
            # Skip coarse detection while tracking is locked on, seeding a tight
            # ROI from the previous confident result (falls back to coarse when
            # tracking is lost). Set False for pure random-access detection.
            "use_prior_roi": True,
        }

    # Base interface

    def get_properties(self):
        return self.properties.copy()

    def update_properties(self, properties):
        for key, value in properties.items():
            if key not in self.properties:
                continue
            expected_type = type(self.properties[key])
            try:
                self.properties[key] = expected_type(value)
            except ValueError as e:
                raise ValueError(
                    f"Value `{repr(value)}` for key `{key}`"
                    f" could not be converted to expected type: {expected_type}"
                ) from e

    def detect(
        self,
        gray_img: np.ndarray,
        color_img: T.Optional[np.ndarray]=None,
        roi: T.Optional[Roi]=None,
        **kwargs
    ) -> T.Dict[str, T.Any]:
        """Detect pupil location in input image.

        Parameters:
            gray_img: input image as 2D numpy array (grayscale)
            color_img (optional): 3D numpy array (BGR)
                will be used to display debug visualizations
            roi (optional): Roi mask for gray_img to speed up detection

        Returns:
            Dictionary with information about the pupil. Keys:
                location (float, float): location of the pupil in image space
                confidence (float): confidence of the algorithm in [0, 1]
                diameter (float): max diameter of the pupil
                ellipse (dict): exact ellipse parameters of the pupil
        """
        cppResultPtr = self.c_detect(gray_img, color_img, roi)
        result = deref(cppResultPtr)

        result_dict = result2D_to_dict(result)
        return result_dict


    cdef shared_ptr[Detector2DResult] c_detect(
        self,
        gray_img: np.ndarray,
        color_img: T.Optional[np.ndarray]=None,
        roi: T.Optional[Roi]=None,
    ):
        image_height, image_width = gray_img.shape

        # cython memory views for accessing the raw data (does not copy)
        # NOTE: [:, ::1] marks the view as c-contiguous
        cdef unsigned char[:, ::1] gray_img_data = gray_img
        cdef unsigned char[:, :, ::1] color_img_data

        cdef Mat frame_image = Mat(image_height, image_width, CV_8UC1, <void *> &gray_img_data[0, 0])
        cdef Mat frameColor

        # not used, but needed for c++ API
        cdef Mat debug_image

        should_visualize = False if color_img is None else True

        if should_visualize:
            color_img_data = color_img
            frameColor = Mat(image_height, image_width, CV_8UC3, <void *> &color_img_data[0, 0, 0])

        cdef int[:, ::1] integral

        # Prior-seeded ROI: while we are locked on, coarse detection (integral +
        # center_surround, ~23% of detect time) is redundant -- the previous
        # confident result already tells us where the pupil is. Seed a tight ROI
        # from it and skip coarse; fall back to a full-frame (coarse) detect if
        # the result comes back weak (e.g. a saccade moved the pupil out of view).
        # An explicit caller-supplied roi always overrides this.
        cdef bint prior_mode = False
        cdef double m
        if roi is None and self.properties['use_prior_roi'] and self.properties['coarse_detection'] and self._have_prev:
            m = self._prev_diam * 0.7
            if m < 55.0:
                m = 55.0
            roi = Roi(
                max(0, <int>(self._prev_cx - m)),
                max(0, <int>(self._prev_cy - m)),
                min(image_width - 1, <int>(self._prev_cx + m)),
                min(image_height - 1, <int>(self._prev_cy + m)),
            )
            prior_mode = True
        elif roi is None:
            roi = Roi.from_rect(0, 0, image_width, image_height)

        cppResultPtr = self._detect_in_roi(roi, frame_image, frameColor, debug_image,
                                           should_visualize, color_img, gray_img)

        # tracking lost in the tight ROI -> relocate with a full-frame coarse pass
        if prior_mode and deref(cppResultPtr).confidence < 0.34:
            roi = Roi.from_rect(0, 0, image_width, image_height)
            cppResultPtr = self._detect_in_roi(roi, frame_image, frameColor, debug_image,
                                               should_visualize, color_img, gray_img)

        # update tracking state from the (absolute-coordinate) result
        if deref(cppResultPtr).confidence > 0.6:
            self._prev_cx = deref(cppResultPtr).ellipse.center[0]
            self._prev_cy = deref(cppResultPtr).ellipse.center[1]
            self._prev_diam = deref(cppResultPtr).ellipse.major_radius * 2.0
            self._have_prev = True
        else:
            self._have_prev = False

        return cppResultPtr

    cdef shared_ptr[Detector2DResult] _detect_in_roi(self, roi, Mat frame_image,
                                                     Mat frameColor, Mat debug_image,
                                                     should_visualize, color_img, gray_img):
        cdef int[:, ::1] integral
        if self.properties['coarse_detection'] and roi.width * roi.height > 320 * 240:
            scale = 2 # half the integral image. boost up integral
            user_roi_image = gray_img[roi.slices]
            integral = cv2.integral(user_roi_image[::scale,::scale])
            coarse_filter_max = self.properties['coarse_filter_max']
            coarse_filter_min = self.properties['coarse_filter_min']
            bounding_box, good_ones, bad_ones = center_surround(
                integral,
                coarse_filter_min / scale,
                coarse_filter_max / scale
            )

            if should_visualize:
                # # draw the candidates
                for v in good_ones:
                    p_x, p_y, w, response = v
                    x = p_x * scale + roi.x_min
                    y = p_y * scale + roi.y_min
                    width = w*scale
                    cv2.rectangle(
                        color_img,
                        (x, y),
                        (x + width, y + width),
                        (255, 255, 0)
                    )

            x1, y1, x2, y2 = bounding_box
            width = x2 - x1
            height = y2 - y1
            roi = Roi.from_rect(
                x=x1 * scale + roi.x_min,
                y=y1 * scale + roi.y_min,
                width=width * scale,
                height=height * scale
            )

        # every coordinates in the result are relative to the current ROI
        return self.thisptr.detect(
            self.properties,
            frame_image,
            frameColor,
            debug_image,
            Rect_[int](roi.x_min, roi.y_min, roi.width, roi.height),
            should_visualize,
            False
        )


cdef object result2D_to_dict(Detector2DResult& result):
    data = {}
    data["ellipse"] = {
        "center": (result.ellipse.center[0], result.ellipse.center[1]),
        "axes": (result.ellipse.minor_radius * 2.0, result.ellipse.major_radius * 2.0),
        "angle": result.ellipse.angle * 180.0 / M_PI - 90.0
    }
    data["diameter"] = max(data["ellipse"]["axes"])
    data["location"] = data["ellipse"]["center"]
    data["confidence"] = result.confidence
    return data
