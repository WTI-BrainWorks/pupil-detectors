"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2019 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

cimport cython

import math


cdef struct point_t :
   int    r
   int    c

cdef struct square_t:
    point_t s
    point_t e
    int a
    float f

cdef struct eye_t:
    square_t outer
    square_t inner
    int w_half
    int w
    int h


@cython.wraparound(False)
@cython.boundscheck(False)
cdef inline int area(int[:,::1] img,point_t size,point_t start,point_t end,point_t offset):
  with cython.boundscheck(False):  # cython bug! ignores @cython.wraparound(False)
      return    img[offset.r + end.r ,offset.c + end.c]\
            + img[offset.r + start.r, offset.c + start.c]\
            - img[offset.r + start.r, offset.c + end.c]\
            - img[offset.r + end.r,offset.c + start.c]

@cython.cdivision(True)
cdef inline eye_t make_eye(int h) nogil:
    cdef int w = 3*h
    cdef eye_t eye
    eye.h = h
    eye.w = w
    eye.outer.s.r = 0
    eye.outer.s.c = 0
    eye.outer.e.r = w
    eye.outer.e.c = w
    eye.inner.s.r = h
    eye.inner.s.c = h
    eye.inner.e.r = h+h
    eye.inner.e.c = h+h
    eye.inner.a = h*h
    eye.outer.a = w*w
    eye.outer.f =  1.0/eye.outer.a
    eye.inner.f =  -1.0/eye.inner.a
    eye.w_half = w/2
    return eye

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline center_surround(int[:,::1] img, int min_w,int max_w):
    cdef int rows = img.shape[0]
    cdef int cols = img.shape[1]
    cdef int min_h = min_w // 3
    cdef int max_h = max_w // 3
    cdef int h=0, i=0, j=0, w=0
    cdef float best_response = -10000
    cdef int h_step = 4
    cdef int step = 5
    cdef float response = 0
    cdef float outer_f = 0, inner_f = 0
    cdef int outer_area, inner_area
    cdef int i_w, i_h, i_2h

    # Raw pointer + row stride for the (C-contiguous) integral image. This avoids
    # the per-access memoryview machinery and the by-value struct copies that the
    # original `area()` helper incurred for every one of the millions of probes.
    cdef int* base = &img[0, 0]
    cdef int stride = cols

    # Ring buffer holding the last 30 "record-improving" candidates, matching the
    # original list semantics (append on each new best, drop oldest beyond 30).
    cdef int CAP = 30
    cdef int rb_j[30]
    cdef int rb_i[30]
    cdef int rb_w[30]
    cdef float rb_r[30]
    cdef int rb_start = 0
    cdef int rb_len = 0
    cdef int slot

    with nogil:
      for h from min_h <= h < max_h by h_step:
        w = 3 * h
        outer_f = 1.0 / (w * w)
        inner_f = -1.0 / (h * h)
        for i from 0 <= i < rows - w by step:
          i_w = (i + w) * stride
          i_h = (i + h) * stride
          i_2h = (i + 2 * h) * stride
          for j from 0 <= j < cols - w by step:
            outer_area = base[i_w + j + w] + base[i * stride + j] \
                       - base[i * stride + j + w] - base[i_w + j]
            inner_area = base[i_2h + j + 2 * h] + base[i_h + j + h] \
                       - base[i_h + j + 2 * h] - base[i_2h + j + h]
            response = outer_f * outer_area + inner_f * inner_area
            if response > best_response:
              best_response = response
              slot = (rb_start + rb_len) % CAP
              if rb_len < CAP:
                rb_len = rb_len + 1
              else:
                rb_start = (rb_start + 1) % CAP
              rb_j[slot] = j
              rb_i[slot] = i
              rb_w[slot] = h * 3
              rb_r[slot] = response

    # Rebuild the Python results list from the ring buffer in chronological order.
    cdef list results = []
    for slot from 0 <= slot < rb_len:
        i = (rb_start + slot) % CAP
        results.append((rb_j[i], rb_i[i], rb_w[i], rb_r[i]))

    #remove results which fully surround others, since we want the smalles ones
    cdef list bad = []
    cdef int x,y,x2,y2,w2
    cdef float response2

    bad = results[:]
    for r in reversed(results):
        x,y,w,response = r
        # filter for response
        if response < best_response*0.4 : #remove it anyway if it's bad
            results.remove(r)
            continue

        for g in bad:
            x2,y2,w2,response2 = g
            if x<x2 and y<y2 and x+w>x2+w2 and y+w>y2+w2: # g is fully included in r
                results.remove(r)
                break

    #calculate bounding box
    cdef int x_b = 0
    cdef int y_b = 0
    cdef int x2_b = 1
    cdef int y2_b = 1

    if len(results) > 0 :
        x_b , y_b  = results[0][:2]
        for v  in results:
            x,y,w,response = v
            x_b  = min(x, x_b)
            y_b  = min(y, y_b)
            if x2_b < x + w:
                x2_b = x+w
            if y2_b < y + w:
                y2_b = y+w

    return  (x_b , y_b, x2_b, y2_b) , results , bad
