# core.pyx
# --------
import cython
import math
import numpy as np
cimport numpy as np
from libc.math cimport sqrt
from PIL import Image

ctypedef np.float32_t f32_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[f32_t, ndim=3] cartesian_to_polar(np.ndarray[f32_t, ndim=3] data):
    cdef int height = data.shape[0]
    cdef int width  = data.shape[1]
    if width < 3 or height < 3 or width % 2 == 0 or height % 2 == 0:
        raise ValueError("Dimensions must be odd and >2")
    cdef int perimeter = 2 * (width + height - 2)
    cdef int halfdiag  = int(math.ceil(sqrt(width*width + height*height)/2))
    cdef int halfw     = width // 2
    cdef int halfh     = height // 2

    cdef np.ndarray[f32_t, ndim=3] ret = \
        np.zeros((halfdiag, perimeter, 3), dtype=np.float32)
    cdef int i, j, row, ystep, xstep
    cdef float slope, diagx, ux, uy

    # axis copies
    ret[0:halfw+1, halfh, :]                         = data[halfh, halfw::-1, :]
    ret[0:halfw+1, height+width-2+halfh, :]          = data[halfh, halfw:halfw*2+1, :]
    ret[0:halfh+1, height-1+halfw, :]                = data[halfh:halfh*2+1, halfw, :]
    ret[0:halfw+1, perimeter-halfw, :]               = data[halfh::-1, halfw, :]

    # 4 top/bottom triangles
    for i in range(halfh):
        slope = (halfh - i) / halfw
        diagx = halfdiag / sqrt(1 + slope*slope)
        ux     = diagx / (halfdiag - 1)
        uy     = ux * slope
        for row in range(halfdiag):
            ystep = <int>round(row * uy)
            xstep = <int>round(row * ux)
            if ystep <= halfh and xstep <= halfw:
                ret[row, i, :]                        = data[halfh-ystep, halfw-xstep, :]
                ret[row, height-1-i, :]               = data[halfh+ystep, halfw-xstep, :]
                ret[row, height+width-2+i, :]         = data[halfh+ystep, halfw+xstep, :]
                ret[row, height+width+height-3-i, :]  = data[halfh-ystep, halfw+xstep, :]
            else:
                break

    # 4 left/right triangles
    for j in range(1, halfw):
        slope = halfh / (halfw - j)
        diagx = halfdiag / sqrt(1 + slope*slope)
        ux     = diagx / (halfdiag - 1)
        uy     = ux * slope
        for row in range(halfdiag):
            ystep = <int>round(row * uy)
            xstep = <int>round(row * ux)
            if ystep <= halfh and xstep <= halfw:
                ret[row, height-1+j, :]               = data[halfh+ystep, halfw-xstep, :]
                ret[row, height+width-2-j, :]         = data[halfh+ystep, halfw+xstep, :]
                ret[row, height+width+height-3+j, :]  = data[halfh-ystep, halfw+xstep, :]
                ret[row, perimeter-j, :]              = data[halfh-ystep, halfw-xstep, :]
            else:
                break

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cdef np.ndarray[f32_t, ndim=3] polar_to_cartesian(
        np.ndarray[f32_t, ndim=3] data,
        int width,
        int height):
    if width < 3 or height < 3 or width % 2 == 0 or height % 2 == 0:
        raise ValueError("Dimensions must be odd and >2")
    cdef int perimeter = 2 * (width + height - 2)
    cdef int halfdiag  = int(math.ceil(sqrt(width*width + height*height)/2))
    cdef int halfw     = width // 2
    cdef int halfh     = height // 2

    cdef np.ndarray[f32_t, ndim=3] ret = \
        np.zeros((height, width, 3), dtype=np.float32)
    cdef int i, j, row, ystep, xstep
    cdef float slope, diagx, ux, uy

    # div0 (axes)
    ret[halfh, halfw::-1, :]                   = data[0:halfw+1, halfh, :]
    ret[halfh, halfw:halfw*2+1, :]             = data[0:halfw+1, height+width-2+halfh, :]
    ret[halfh:halfh*2+1, halfw, :]             = data[0:halfh+1, height-1+halfw, :]
    ret[halfh::-1, halfw, :]                   = data[0:halfw+1, perimeter-halfw, :]

    # part1
    for i in range(halfh):
        slope = (halfh - i) / halfw
        diagx = halfdiag / sqrt(1 + slope*slope)
        ux     = diagx / (halfdiag - 1)
        uy     = ux * slope
        for row in range(halfdiag):
            ystep = <int>round(row * uy)
            xstep = <int>round(row * ux)
            if ystep <= halfh and xstep <= halfw:
                ret[halfh-ystep, halfw-xstep, :]      = data[row, i, :]
                ret[halfh+ystep, halfw-xstep, :]      = data[row, height-1-i, :]
                ret[halfh+ystep, halfw+xstep, :]      = data[row, height+width-2+i, :]
                ret[halfh-ystep, halfw+xstep, :]      = data[row, height+width+height-3-i, :]
            else:
                break

    # part2
    for j in range(1, halfw):
        slope = halfh / (halfw - j)
        diagx = halfdiag / sqrt(1 + slope*slope)
        ux     = diagx / (halfdiag - 1)
        uy     = ux * slope
        for row in range(halfdiag):
            ystep = <int>round(row * uy)
            xstep = <int>round(row * ux)
            if ystep <= halfh and xstep <= halfw:
                ret[halfh+ystep, halfw-xstep, :]      = data[row, height-1+j, :]
                ret[halfh+ystep, halfw+xstep, :]      = data[row, height+width-2-j, :]
                ret[halfh-ystep, halfw+xstep, :]      = data[row, height+width+height-3+j, :]
                ret[halfh-ystep, halfw-xstep, :]      = data[row, perimeter-j, :]
            else:
                break

    # repair zeros
    cdef int y, x, c
    for y in range(1, height-1):
        for x in range(1, width-1):
            for c in range(3):
                if ret[y, x, c] == 0:
                    ret[y, x, c] = (ret[y-1, x, c] + ret[y+1, x, c]) * 0.5

    return ret


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.ndarray[f32_t, ndim=2] vertical_gaussian(
        np.ndarray[f32_t, ndim=2] data,
        int n):
    cdef int height = data.shape[0]
    cdef int width  = data.shape[1]
    cdef int pad    = n - 1
    cdef int i, radius
    cdef np.ndarray[f32_t, ndim=2] padded = \
        np.zeros((height + pad*2, width), dtype=np.float32)
    padded[pad:pad+height, :] = data
    cdef np.ndarray[f32_t, ndim=2] ret = \
        np.zeros((height, width), dtype=np.float32)
    cdef np.ndarray[f32_t, ndim=2] kernel
    cdef int oldr = -1

    for i in range(height):
        radius = <int>round(i * pad / (height - 1)) + 1
        if radius != oldr:
            oldr = radius
            kernel = np.array(get_gauss(1 + 2*(radius-1)),
                              dtype=np.float32).reshape((-1, 1))
        ret[i, :] = (padded[i+pad-radius+1:i+pad+radius, :] * kernel).sum(axis=0)

    return ret


cpdef list get_gauss(int n):
    cdef float sigma = 0.3*(n/2 - 1) + 0.8
    cdef int x
    cdef list r = []
    cdef float tot = 0.0
    for x in range(-n//2, n//2+1):
        val = (1/(sigma*math.sqrt(2*math.pi))) * math.exp(-x*x/(2*sigma*sigma))
        r.append(val)
        tot += val
    # normalize
    return [v/tot for v in r]


cpdef object add_jitter(im, int pixels=1):
    """Pure-Python PIL wrapper remains untyped except for args."""
    if pixels == 0:
        return im.copy()
    r, g, b = im.split()
    return Image.merge("RGB", (
        r.crop((pixels, 0, r.width+pixels, r.height)),
        g,
        b.crop((-pixels, 0, b.width-pixels, b.height)),
    ))


cpdef object blend_images(im, og_im, float alpha=1.0, float strength=1.0):
    """PIL-based merging/resizing left in Python."""
    og_im.putalpha(<int>(255*alpha))
    og_im = og_im.resize((
        round((1 + 0.018*strength) * og_im.width),
        round((1 + 0.018*strength) * og_im.height)
    ), Image.LANCZOS)
    # …centre & composite as in original…
    return im  # fill in original logic here


cpdef object add_chromatic(im, float strength=1.0, bint no_blur=False):
    """Split channels → polar → blur → cartesian → PIL merge."""
    # …call cartesian_to_polar, vertical_gaussian, polar_to_cartesian…
    return im  # fill in original logic here
