import math

import numpy as np
from scipy import ndimage


def rgb_to_gray(image):
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def rgb_to_hsv(image):
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    maxc = np.max(image, axis=-1)
    minc = np.min(image, axis=-1)
    delta = maxc - minc
    h = np.zeros_like(maxc)
    mask = delta > 1e-8
    rmask = mask & (maxc == r)
    gmask = mask & (maxc == g)
    bmask = mask & (maxc == b)
    h[rmask] = ((g[rmask] - b[rmask]) / delta[rmask]) % 6.0
    h[gmask] = ((b[gmask] - r[gmask]) / delta[gmask]) + 2.0
    h[bmask] = ((r[bmask] - g[bmask]) / delta[bmask]) + 4.0
    h /= 6.0
    s = np.zeros_like(maxc)
    s[maxc > 1e-8] = delta[maxc > 1e-8] / maxc[maxc > 1e-8]
    v = maxc
    return np.stack([h, s, v], axis=-1).astype(np.float32)


def l1_normalize(vec):
    vec = np.asarray(vec, dtype=np.float32)
    total = float(vec.sum())
    if total > 0:
        vec = vec / total
    return vec


def gch(image):
    hsv = rgb_to_hsv(image)
    hb = np.minimum((hsv[..., 0] * 16).astype(np.int32), 15)
    sb = np.minimum((hsv[..., 1] * 4).astype(np.int32), 3)
    vb = np.minimum((hsv[..., 2] * 4).astype(np.int32), 3)
    idx = hb * 16 + sb * 4 + vb
    hist = np.bincount(idx.ravel(), minlength=256).astype(np.float32)
    return l1_normalize(hist)


def lbp(image):
    gray = rgb_to_gray(image)
    center = gray[1:-1, 1:-1]
    neighbors = [
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    ]
    codes = np.zeros_like(center, dtype=np.uint8)
    for bit, neigh in enumerate(neighbors):
        codes |= ((neigh >= center).astype(np.uint8) << bit)
    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
    return l1_normalize(hist)


def glcm(image):
    gray = rgb_to_gray(image)
    levels = 32
    q = np.minimum((gray * levels).astype(np.int32), levels - 1)
    offsets = [
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, 2),
        (-2, 2),
        (-2, 0),
        (-2, -2),
        (0, 3),
        (-3, 3),
        (-3, 0),
        (-3, -3),
    ]
    props = []
    ii, jj = np.indices((levels, levels))
    for dy, dx in offsets:
        a, b = _offset_views(q, dy, dx)
        mat = np.bincount((a.ravel() * levels + b.ravel()), minlength=levels * levels).reshape(levels, levels)
        mat = mat.astype(np.float64)
        total = mat.sum()
        if total == 0:
            props.append([0.0, 0.0, 0.0, 0.0, 0.0])
            continue
        p = mat / total
        contrast = np.sum((ii - jj) ** 2 * p)
        homogeneity = np.sum(p / (1.0 + np.abs(ii - jj)))
        energy = np.sqrt(np.sum(p ** 2))
        mui = np.sum(ii * p)
        muj = np.sum(jj * p)
        sdi = np.sqrt(np.sum(((ii - mui) ** 2) * p))
        sdj = np.sqrt(np.sum(((jj - muj) ** 2) * p))
        corr = 0.0 if sdi * sdj == 0 else np.sum((ii - mui) * (jj - muj) * p) / (sdi * sdj)
        ent = -np.sum(p[p > 0] * np.log2(p[p > 0]))
        props.append([contrast, homogeneity, energy, corr, ent])
    arr = np.asarray(props, dtype=np.float32)
    return np.concatenate([arr.mean(axis=0), arr.std(axis=0)]).astype(np.float32)


def _offset_views(q, dy, dx):
    h, w = q.shape
    y0a = max(0, -dy)
    y1a = min(h, h - dy)
    x0a = max(0, -dx)
    x1a = min(w, w - dx)
    y0b = y0a + dy
    y1b = y1a + dy
    x0b = x0a + dx
    x1b = x1a + dx
    return q[y0a:y1a, x0a:x1a], q[y0b:y1b, x0b:x1b]


def hog(image):
    gray = rgb_to_gray(image)
    gray = ndimage.zoom(gray, (128 / gray.shape[0], 128 / gray.shape[1]), order=1)
    gx = ndimage.sobel(gray, axis=1, mode="reflect")
    gy = ndimage.sobel(gray, axis=0, mode="reflect")
    mag = np.hypot(gx, gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) % 180.0)
    cell = 16
    bins = 9
    hist = np.zeros((8, 8, bins), dtype=np.float32)
    bin_idx = np.minimum((ang / 20.0).astype(np.int32), bins - 1)
    for cy in range(8):
        for cx in range(8):
            ys = slice(cy * cell, (cy + 1) * cell)
            xs = slice(cx * cell, (cx + 1) * cell)
            hist[cy, cx] = np.bincount(bin_idx[ys, xs].ravel(), weights=mag[ys, xs].ravel(), minlength=bins)
    blocks = []
    for cy in range(7):
        for cx in range(7):
            block = hist[cy : cy + 2, cx : cx + 2].ravel()
            norm = np.sqrt(np.sum(block ** 2) + 1e-6)
            blocks.append(block / norm)
    return np.concatenate(blocks).astype(np.float32)


def correlogram(image):
    hsv = rgb_to_hsv(image)
    q = np.minimum((hsv[..., 0] * 16).astype(np.int32), 15)
    feats = []
    for dist in (1, 3, 5):
        counts = np.zeros(16, dtype=np.float64)
        totals = np.zeros(16, dtype=np.float64)
        for dy, dx in ((0, dist), (dist, 0), (dist, dist), (dist, -dist)):
            a, b = _offset_views(q, dy, dx)
            same = a == b
            totals += np.bincount(a.ravel(), minlength=16)
            counts += np.bincount(a[same].ravel(), minlength=16)
        vals = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
        feats.append(vals.astype(np.float32))
    return np.concatenate(feats).astype(np.float32)


def gabor(image):
    gray = rgb_to_gray(image)
    gray = ndimage.zoom(gray, (128 / gray.shape[0], 128 / gray.shape[1]), order=1)
    feats = []
    for freq in (0.05, 0.1, 0.2, 0.3, 0.4):
        sigma = max(2.0, 0.56 / freq)
        radius = int(min(10, max(4, math.ceil(2.5 * sigma))))
        y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
        for theta in np.linspace(0, np.pi, 8, endpoint=False):
            xr = x * np.cos(theta) + y * np.sin(theta)
            yr = -x * np.sin(theta) + y * np.cos(theta)
            envelope = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2))
            kernel = envelope * np.cos(2 * np.pi * freq * xr)
            kernel -= kernel.mean()
            denom = np.sqrt(np.sum(kernel ** 2))
            if denom > 0:
                kernel /= denom
            response = ndimage.convolve(gray, kernel.astype(np.float32), mode="reflect")
            feats.extend([float(response.mean()), float(response.std()), float(np.mean(response ** 2))])
    return np.asarray(feats, dtype=np.float32)


DESCRIPTOR_FUNCS = {
    "gch": gch,
    "lbp": lbp,
    "glcm": glcm,
    "hog": hog,
    "correlogram": correlogram,
    "gabor": gabor,
}

