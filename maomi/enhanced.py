"""Rana Mode: full-field cubic PSF imaging, independent of native MATLAB assembly.

Every source (detection) or scan position (excitation) uses its own local kernel.
XY field variation supports either XY or XYZ offset kernels. No average-PSF
fallback, extrapolation, automatic normalization, or automatic multiphoton power.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
from scipy import fft
from scipy.linalg import lu_factor, lu_solve
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class FullFieldPSF:
    """Effective intensity kernels with their spatial gains retained.

    Axes are field_X, field_Y, kernel_X, kernel_Y[, kernel_Z]. Kernel origin is
    shape // 2; coordinates and voxel pitches are in specimen micrometres.
    operator must explicitly be 'detection' or 'excitation'. For two-photon
    excitation, supply an already effective intensity-squared kernel with the
    desired power calibration. The renderer never squares or normalizes it.
    """

    kernels: np.ndarray
    x_um: np.ndarray
    y_um: np.ndarray
    voxel_size_um: tuple
    wavelength_nm: float
    operator: str
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if np.iscomplexobj(self.kernels):
            raise ValueError("Supply effective real intensity, not a complex optical field")
        a = np.array(self.kernels, dtype=np.float64, copy=True)
        coords = [np.array(c, dtype=float, copy=True) for c in (self.x_um, self.y_um)]
        for c in coords:
            if c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all() or (np.diff(c) <= 0).any():
                raise ValueError("Full-field PSFs require at least two increasing finite nodes on EACH XY axis")
        if (a.ndim not in (4, 5) or a.shape[:2] != tuple(len(c) for c in coords)
                or min(a.shape[2:]) < 1 or not np.isfinite(a).all() or (a < 0).any()):
            raise ValueError("Supply nonnegative finite kernels with axes field_X,field_Y,kernel_X,kernel_Y[,kernel_Z]")
        gains = a.sum(axis=tuple(range(2, a.ndim)))
        if not np.isfinite(gains).all() or (gains <= 0).any():
            raise ValueError("Every local PSF must have positive finite energy")
        pitch = tuple(float(v) for v in self.voxel_size_um)
        if len(pitch) != a.ndim - 2 or not np.isfinite(pitch).all() or min(pitch) <= 0:
            raise ValueError("Specify one positive voxel pitch per kernel dimension")
        if not np.isfinite(self.wavelength_nm) or self.wavelength_nm <= 0:
            raise ValueError("wavelength_nm must be positive and finite")
        if self.operator not in ("detection", "excitation"):
            raise ValueError("Declare operator='detection' or operator='excitation' explicitly")
        meta = json.loads(json.dumps(self.provenance, allow_nan=False))
        for value in [a, *coords]:
            value.setflags(write=False)
        object.__setattr__(self, "kernels", a)
        object.__setattr__(self, "x_um", coords[0])
        object.__setattr__(self, "y_um", coords[1])
        object.__setattr__(self, "voxel_size_um", pitch)
        object.__setattr__(self, "provenance", meta)

    @classmethod
    def from_psf_field(cls, psf_field, *, operator, gains=None):
        """Adapt a normalized 2-D PSFField; restore gains explicitly if available.

        operator='excitation' asserts these are already effective excitation
        shapes; this adapter cannot infer or convert their physical meaning.
        """
        from .psf import PSFField
        if not isinstance(psf_field, PSFField):
            raise TypeError("Expected PSFField, not a single shared PSF")
        shape = psf_field.intensity.shape[:2]
        g = np.ones(shape) if gains is None else np.asarray(gains, dtype=float)
        if g.shape != shape or not np.isfinite(g).all() or (g <= 0).any():
            raise ValueError("gains must contain a positive finite value per field node")
        return cls(psf_field.intensity * g[..., None, None], psf_field.x_um,
                   psf_field.y_um, psf_field.pixel_size_um, psf_field.wavelength_nm,
                   operator, {"source": psf_field.provenance,
                              "gain_source": "unit gains" if gains is None else "explicit node gains"})

    @classmethod
    def from_provider(cls, provider, x_um, y_um, *, voxel_size_um, wavelength_nm,
                      operator, provenance=None):
        """Sample a callable returning effective arrays, without renormalization."""
        x, y = np.asarray(x_um, dtype=float), np.asarray(y_um, dtype=float)
        for c in (x, y):
            if c.ndim != 1 or len(c) < 2 or not np.isfinite(c).all() or (np.diff(c) <= 0).any():
                raise ValueError("Provider grid needs at least two increasing finite nodes per axis")
        arrays = [[np.asarray(provider(float(xx), float(yy))) for yy in y] for xx in x]
        return cls(np.array(arrays), x, y, voxel_size_um, wavelength_nm, operator, provenance or {})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, schema="maomi.full_field_psf.v1", kernels=self.kernels,
                            x_um=self.x_um, y_um=self.y_um, voxel_size_um=self.voxel_size_um,
                            wavelength_nm=self.wavelength_nm, operator=self.operator,
                            provenance=json.dumps(self.provenance, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            if str(d["schema"]) != "maomi.full_field_psf.v1":
                raise ValueError("Unsupported full-field PSF schema")
            return cls(d["kernels"], d["x_um"], d["y_um"], tuple(d["voxel_size_um"]),
                       float(d["wavelength_nm"]), str(d["operator"]), json.loads(str(d["provenance"])))


class CubicFullFieldRenderer:
    """Affine-augmented r^3 interpolation of complete local effective kernels.

    An exact basis decomposition avoids one convolution per image pixel; it
    does not approximate the operator by an averaged PSF. Interpolated kernels
    are checked at every XY pixel center before use. Material negative values
    are rejected, because cubic RBF weights need not be positive. Coverage is
    required out to the FOV edges. Outside-image fluorescence is zero.

    3-D kernels in this class vary with XY. DepthPSFBank and
    DepthFullFieldRenderer add absolute focal-Z variation. Native movie/tail
    workflows use the standard mode.
    """

    def __init__(self, psf_field, shape, *, voxel_size_um, origin_um=(0., 0.), device="cpu"):
        if not isinstance(psf_field, FullFieldPSF):
            raise TypeError("Rana Mode requires FullFieldPSF; a shared PSF is not accepted")
        dims = psf_field.kernels.ndim - 2
        if len(shape) != dims or any(isinstance(n, bool) or int(n) != n or n < 1 for n in shape):
            raise ValueError("Image shape must contain positive integers matching kernel dimensions")
        self.shape = tuple(int(n) for n in shape)
        pitch = np.asarray(voxel_size_um, dtype=float)
        origin = np.asarray(origin_um, dtype=float)
        if (pitch.shape != (dims,) or not np.isfinite(pitch).all() or (pitch <= 0).any()
                or not np.allclose(pitch, psf_field.voxel_size_um, rtol=1e-12, atol=0)):
            raise ValueError("Image and kernel voxel pitches must match; resample explicitly")
        if origin.shape != (2,) or not np.isfinite(origin).all():
            raise ValueError("origin_um must contain two finite XY lower-edge coordinates")
        lower = np.array([psf_field.x_um[0], psf_field.y_um[0]])
        upper = np.array([psf_field.x_um[-1], psf_field.y_um[-1]])
        end = origin + np.array(self.shape[:2]) * pitch[:2]
        if (origin < lower - 1e-12).any() or (end > upper + 1e-12).any():
            raise ValueError("PSF field must cover the entire FOV including edges; extrapolation is forbidden")
        if device != "cpu" and not str(device).startswith("cuda"):
            raise ValueError("device must be cpu or cuda[:index]")
        self.field = psf_field
        self.device = device
        self._lower, self._span = lower, upper - lower
        self._nodes = (np.stack(np.meshgrid(psf_field.x_um, psf_field.y_um, indexing="ij"), -1)
                       .reshape(-1, 2) - lower) / self._span
        n = len(self._nodes)
        poly = np.c_[np.ones(n), self._nodes]
        system = np.block([[cdist(self._nodes, self._nodes)**3, poly],
                           [poly.T, np.zeros((3, 3))]])
        self.kernel_shape = psf_field.kernels.shape[2:]
        rhs = np.r_[psf_field.kernels.reshape(n, -1), np.zeros((3, np.prod(self.kernel_shape)))]
        # Exact input symmetry, not a shared-PSF approximation: when every
        # supplied XY kernel is bitwise identical, only the constant basis is
        # nonzero. This avoids roundoff coefficients and redundant FFTs/checks.
        self._uniform_xy = np.array_equal(psf_field.kernels,
                                         np.broadcast_to(psf_field.kernels[0, 0], psf_field.kernels.shape))
        if self._uniform_xy:
            self._coeff = np.zeros_like(rhs)
            self._coeff[n] = psf_field.kernels[0, 0].ravel()
        else:
            self._coeff = lu_solve(lu_factor(system), rhs)
        if not np.isfinite(self._coeff).all():
            raise ValueError("Ill-conditioned cubic PSF field")
        axes = [origin[i] + (np.arange(self.shape[i]) + .5) * pitch[i] for i in range(2)]
        self._pixels = ((np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 2)
                         - lower) / self._span)
        # Bound query-kernel scratch space to approximately 16 MiB.
        self._chunk = max(1, min(1024, 2_000_000 // (self._coeff.shape[0] + self._coeff.shape[1])))
        worst = np.inf
        minimum_gain = np.inf
        blocks = [psf_field.kernels[0, 0].reshape(1, -1)] if self._uniform_xy else (
            self._basis(self._pixels[start:start+self._chunk]) @ self._coeff
            for start in range(0, len(self._pixels), self._chunk))
        for local in blocks:
            worst = min(worst, float(local.min()))
            minimum_gain = min(minimum_gain, float(local.sum(1).min()))
            self._check_local(local)
        self.validation = {"checked_xy_positions": len(self._pixels),
                           "minimum_interpolated_value": worst,
                           "minimum_local_gain": minimum_gain,
                           "exact_uniform_xy": self._uniform_xy,
                           "shared_psf_fallback": False}
        self._fft_shape = tuple(fft.next_fast_len(n + k - 1) for n, k in zip(self.shape, self.kernel_shape))
        centers = [k // 2 if psf_field.operator == "detection" else k - 1 - k // 2
                   for k in self.kernel_shape]
        self._crop = tuple(slice(c, c + n) for c, n in zip(centers, self.shape))
        if device != "cpu":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA PyTorch is required for device='cuda'")
            self._torch = torch

    def _basis(self, q):
        return np.c_[cdist(q, self._nodes)**3, np.ones(len(q)), q]

    @staticmethod
    def _check_local(local):
        scale = np.max(np.abs(local), axis=1)
        if (not np.isfinite(local).all() or (local.sum(1) <= 0).any()
                or (local.min(1) < -1e-10 * scale).any()):
            raise ValueError("Cubic interpolation produced nonphysical local PSFs; "
                             "refine or revise the supplied field. No shared-PSF fallback is allowed")

    def local_psfs(self, xy_um):
        """Inspect interpolated effective kernels at covered physical positions."""
        xy = np.asarray(xy_um, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
            raise ValueError("xy_um must be a finite (N, 2) array")
        q = (xy - self._lower) / self._span
        if (q < 0).any() or (q > 1).any():
            raise ValueError("PSF extrapolation is forbidden")
        if len(q) == 0:
            return np.empty((0, *self.kernel_shape))
        blocks = []
        for start in range(0, len(q), self._chunk):
            local = self._basis(q[start:start+self._chunk]) @ self._coeff
            self._check_local(local)
            blocks.append(local.reshape(-1, *self.kernel_shape))
        return np.concatenate(blocks)

    def _weight(self, index):
        if index < len(self._nodes):
            values = np.linalg.norm(self._pixels - self._nodes[index], axis=1)**3
        elif index == len(self._nodes):
            values = np.ones(len(self._pixels))
        else:
            values = self._pixels[:, index - len(self._nodes) - 1]
        return values.reshape(*self.shape[:2], *((1,) if len(self.shape) == 3 else ()))

    def __call__(self, density):
        if np.iscomplexobj(density):
            raise ValueError("Fluorescence density must be real")
        a = np.asarray(density, dtype=float)
        if a.shape != self.shape or not np.isfinite(a).all() or (a < 0).any():
            raise ValueError("Density must be finite, nonnegative and match the prepared shape")
        cuda = self.device != "cpu"
        if cuda:
            t = self._torch
            convert = lambda x: t.as_tensor(np.ascontiguousarray(x), dtype=t.float64, device=self.device)
            forward = lambda x: t.fft.rfftn(x, s=self._fft_shape)
            inverse = lambda x: t.fft.irfftn(x, s=self._fft_shape)
            source = convert(a)
        else:
            convert = np.asarray
            forward = lambda x: fft.rfftn(x, s=self._fft_shape, workers=1)
            inverse = lambda x: fft.irfftn(x, s=self._fft_shape, workers=1)
            source = a
        excitation = self.field.operator == "excitation"
        spectrum = forward(source) if excitation else None
        accumulated = None
        for i, flat in enumerate(self._coeff):
            kernel = flat.reshape(self.kernel_shape)
            if not np.any(kernel):
                continue
            weight = convert(self._weight(i))
            if excitation:
                response = inverse(spectrum * forward(convert(np.flip(kernel))))[self._crop] * weight
            else:
                response = forward(source * weight) * forward(convert(kernel))
            accumulated = response if accumulated is None else accumulated + response
        result = accumulated if excitation else inverse(accumulated)[self._crop]
        if cuda:
            result = result.cpu().numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite full-field image")
        scale = max(float(np.max(np.abs(result))), np.finfo(float).tiny)
        if float(result.min()) < -1e-9 * scale:
            raise FloatingPointError("Full-field image has material negative intensity")
        return np.maximum(result, 0.)  # Only floating-point cancellation, after checks.
