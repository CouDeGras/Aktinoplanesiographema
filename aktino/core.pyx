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

    # ───────────── axis copies (contiguous via .copy()) ─────────────
    # horizontal axis – length = halfw + 1
    ret[0:halfw+1,  halfh,                :] = data[halfh,           halfw::-1,      :].copy()
    ret[0:halfw+1,  height+width-2+halfh, :] = data[halfh,           halfw:halfw*2+1, :].copy()

    # vertical axis – length = halfh + 1
    ret[0:halfh+1,  height-1+halfw,       :] = data[halfh:halfh*2+1, halfw,          :].copy()
    ret[0:halfh+1,  perimeter-halfw,      :] = data[halfh::-1,       halfw,          :].copy()



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
    cdef int half = n // 2                 # centred half-width (positive)
    cdef int x
    cdef list r = []
    cdef float tot = 0.0
    for x in range(-half, half + 1):       # ← correct, matches original Python
        val = (1/(sigma*math.sqrt(2*math.pi))) * math.exp(-x*x/(2*sigma*sigma))
        r.append(val)
        tot += val
    return [v / tot for v in r]



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


# ---------------------------------------------------------------------------
# 🔄  Replace the current stub of blend_images with the full implementation
# ---------------------------------------------------------------------------
cpdef object blend_images(im, og_im, float alpha=1.0, float strength=1.0):
    """
    Alpha–blends `og_im` on top of `im`, after a slight scale-up that
    matches the chromatic expansion factor. 100 % Python/PIL code – no
    performance concerns here.
    """
    if alpha <= 0.0:
        # nothing to blend
        return im

    # 1) add alpha channel to original image
    og_im = og_im.copy()
    og_im.putalpha(<int>(255 * alpha))

    # 2) enlarge the overlay using the same 0.018·strength scale factor
    og_im = og_im.resize(
        (
            round((1.0 + 0.018 * strength) * og_im.width),
            round((1.0 + 0.018 * strength) * og_im.height),
        ),
        Image.LANCZOS,          # ≈ PIL.Image.ANTIALIAS in modern Pillow
    )

    # 3) centre-crop overlay so it matches `im`'s size
    cdef int hdiff = (og_im.height - im.height) // 2
    cdef int wdiff = (og_im.width  - im.width)  // 2
    og_im = og_im.crop(
        (wdiff, hdiff, wdiff + im.width, hdiff + im.height)
    )

    # 4) composite
    im_rgba = im.convert("RGBA")
    base    = Image.new("RGBA", im.size)
    base    = Image.alpha_composite(base, im_rgba)
    base    = Image.alpha_composite(base, og_im)

    return base.convert("RGB")



cpdef object add_chromatic(im, float strength=1.0, bint no_blur=False):
    """
    Radial chromatic-aberration effect – Cython version.
    """
    # --------  NEW SAFETY CROPPING  -----------------------------------
    # 1. declare the ints at the top level
    cdef int new_w, new_h

    if (im.width % 2 == 0) or (im.height % 2 == 0):
        new_w = im.width  - (1 if im.width  % 2 == 0 else 0)
        new_h = im.height - (1 if im.height % 2 == 0 else 0)
        im = im.crop((0, 0, new_w, new_h))
    # ------------------------------------------------------------------
    # 1) Split channels and cast to float32 NumPy arrays
    # ------------------------------------------------------------------
    r, g, b = im.split()
    cdef np.ndarray[f32_t, ndim=2] r_np = np.asarray(r, dtype=np.float32)
    cdef np.ndarray[f32_t, ndim=2] g_np = np.asarray(g, dtype=np.float32)
    cdef np.ndarray[f32_t, ndim=2] b_np = np.asarray(b, dtype=np.float32)

    # ------------------------------------------------------------------
    # 2) Top-level typed variables  (MUST be one per line)
    # ------------------------------------------------------------------
    cdef np.ndarray[f32_t, ndim=3] rgb = None
    cdef np.ndarray[f32_t, ndim=3] poles = None
    cdef np.ndarray[f32_t, ndim=2] r_pol
    cdef np.ndarray[f32_t, ndim=2] g_pol
    cdef np.ndarray[f32_t, ndim=2] b_pol
    cdef np.ndarray[f32_t, ndim=3] cartes = None
    cdef float blur
    cdef int   br


    cdef object r_final = r     # default in case no_blur is True
    cdef object g_final = g
    cdef object b_final = b

    # ------------------------------------------------------------------
    # 3) Heavy maths only if we really blur
    # ------------------------------------------------------------------
    if not no_blur:
        # stack channels → polar space
        rgb = np.ascontiguousarray(
            np.stack([r_np, g_np, b_np], axis=-1), dtype=np.float32
        )

        poles = cartesian_to_polar(rgb)
        r_pol, g_pol, b_pol = poles[:, :, 0], poles[:, :, 1], poles[:, :, 2]

        # adaptive blur radius
        blur = (im.width + im.height - 2) / 100.0 * strength
        br   = <int>round(blur)

        if br > 0:
            r_pol = vertical_gaussian(r_pol, br)
            g_pol = vertical_gaussian(g_pol, <int>round(blur * 1.2))
            b_pol = vertical_gaussian(b_pol, <int>round(blur * 1.4))

        cartes = polar_to_cartesian(
            np.stack([r_pol, g_pol, b_pol], axis=-1),
            im.width, im.height
        )

        cartes = np.clip(cartes, 0.0, 255.0)
        r_final = Image.fromarray(cartes[:, :, 0].astype(np.uint8), "L")
        g_final = Image.fromarray(cartes[:, :, 1].astype(np.uint8), "L")
        b_final = Image.fromarray(cartes[:, :, 2].astype(np.uint8), "L")

    # ------------------------------------------------------------------
    # 4) Channel expansion
    # ------------------------------------------------------------------
    g_final = g_final.resize(
        (
            round((1.0 + 0.018 * strength) * r.width),
            round((1.0 + 0.018 * strength) * r.height),
        ),
        Image.LANCZOS,
    )
    b_final = b_final.resize(
        (
            round((1.0 + 0.044 * strength) * r.width),
            round((1.0 + 0.044 * strength) * r.height),
        ),
        Image.LANCZOS,
    )

    # ------------------------------------------------------------------
    # 5) Centre-align & merge
    # ------------------------------------------------------------------
    cdef int rhdiff = (b_final.height - r_final.height) // 2
    cdef int rwdiff = (b_final.width  - r_final.width)  // 2
    cdef int ghdiff = (b_final.height - g_final.height) // 2
    cdef int gwdiff = (b_final.width  - g_final.width)  // 2

    merged = Image.merge(
        "RGB",
        (
            r_final.crop((-rwdiff, -rhdiff,
                          b_final.width - rwdiff,
                          b_final.height - rhdiff)),
            g_final.crop((-gwdiff, -ghdiff,
                          b_final.width - gwdiff,
                          b_final.height - ghdiff)),
            b_final,
        ),
    )

    # restore original odd dimensions
    return merged.crop(
        (
            rwdiff, rhdiff,
            rwdiff + r_final.width,
            rhdiff + r_final.height,
        )
    )
