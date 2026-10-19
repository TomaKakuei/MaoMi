"""Reusable CPU image formation, separate from providers and detector noise.

For emitting position s and output position r, h(r-s; s) is interpolated at s.
image = sum_k convolution(density * weight_k, kernel_k). Weighting blurred output
images instead would implement a different operator.
"""
import numpy as np
from scipy.fft import rfftn, irfftn, next_fast_len
from .psf import SampledPSF, PSFField


def _axis_weights(nodes, positions):
    if len(nodes) == 1:
        return np.ones((1, len(positions)))
    s = np.clip(positions, nodes[0], nodes[-1])
    left = np.clip(np.searchsorted(nodes, s, side="right")-1, 0, len(nodes)-2)
    fraction = (s-nodes[left])/(nodes[left+1]-nodes[left])
    w = np.zeros((len(nodes), len(positions)))
    columns = np.arange(len(positions))
    w[left, columns] = 1.-fraction
    w[left+1, columns] = fraction
    return w


class PSFRenderer:
    """Compile kernel FFTs once and reuse across frames of the same shape.

    CPU float64; zero padding; crop origin at kernel.shape // 2, including even
    kernels (the existing MaoMi/MATLAB convention). origin_um is the lower edge
    of pixel (0,0); centers add half a pitch. Boundary light may leave the image.
    Memory scales with active field nodes times padded image size.
    """
    def __init__(self, psf, shape, *, pixel_size_um, origin_um=(0., 0.)):
        if not isinstance(psf, (SampledPSF, PSFField)):
            raise TypeError("psf must be SampledPSF or PSFField")
        if len(shape) != 2 or any(int(n) != n or n < 1 for n in shape):
            raise ValueError("shape must contain two positive integer dimensions")
        pitch, origin = np.asarray(pixel_size_um, float), np.asarray(origin_um, float)
        if pitch.shape != (2,) or not np.isfinite(pitch).all() or (pitch <= 0).any():
            raise ValueError("pixel_size_um must contain two finite positive values")
        if not np.allclose(pitch, psf.pixel_size_um, rtol=1e-10, atol=1e-12):
            raise ValueError("Scene and PSF sampling must match; call resample_psf explicitly")
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("origin_um must contain two finite values")
        self.shape = tuple(int(n) for n in shape)
        self.pixel_size_um = tuple(pitch)
        self.origin_um = tuple(origin)
        kernel_shape = psf.intensity.shape[-2:]
        self.fft_shape = tuple(next_fast_len(n+k-1, real=True) for n, k in zip(self.shape, kernel_shape))
        self._crop = tuple(slice(k//2, k//2+n) for n, k in zip(self.shape, kernel_shape))
        if isinstance(psf, SampledPSF):
            kernels = psf.intensity[None, None, :, :]
            wx, wy = np.ones((1, self.shape[0])), np.ones((1, self.shape[1]))
        else:
            kernels = psf.intensity
            wx, wy = [_axis_weights(nodes, o+(np.arange(n)+.5)*p)
                      for nodes, o, n, p in zip((psf.x_um, psf.y_um), origin, self.shape, pitch)]
        self._terms = []
        for i, xweight in enumerate(wx):
            for j, yweight in enumerate(wy):
                if xweight.any() and yweight.any():
                    spectrum = rfftn(kernels[i, j], s=self.fft_shape, workers=1)
                    spectrum.setflags(write=False)
                    self._terms.append((xweight[:, None]*yweight[None, :], spectrum))

    @property
    def active_kernels(self):
        return len(self._terms)

    def __call__(self, density):
        a = np.asarray(density, dtype=np.float64)
        if a.shape != self.shape or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("Density must match the renderer shape and be finite and nonnegative")
        total = None
        for weights, kernel in self._terms:
            term = rfftn(a*weights, s=self.fft_shape, workers=1)
            term *= kernel
            if total is None:
                total = term
            else:
                total += term
        result = irfftn(total, s=self.fft_shape, workers=1)[self._crop]
        if result.min() < -1e-10*max(1., float(a.max())):
            raise ValueError("Convolution produced a material negative intensity")
        return np.maximum(result, 0.)


def render_image(density, psf, *, pixel_size_um, origin_um=(0., 0.)):
    """One-shot API; use PSFRenderer to reuse transforms across frames."""
    return PSFRenderer(psf, np.shape(density), pixel_size_um=pixel_size_um,
                       origin_um=origin_um)(density)
