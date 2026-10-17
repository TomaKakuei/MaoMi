"""Preserved field primitives; propagation lives in optics/backends."""

import numpy as np
import torch
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
import typing


@dataclass
class VolParams:
    vol_sz: np.ndarray = field(default_factory=lambda: np.array([100, 100, 30], dtype=np.float32))
    vres: float = 2.0
    vol_depth: float = 100.0
    vasc_sz: typing.Optional[np.ndarray] = None
    verbose: int = 1


def masked_3DGP_test(grid_sz, l_scale, p_scale, mu=0, bin_mask=1, l_weights=None, device="cpu"):
    if l_weights is None:
        l_weights = torch.ones(l_scale.shape[0], device=device)

    grid_sz = list(int(x) for x in grid_sz)
    wmx = np.pi / 2
    grid_x = (
        torch.linspace(-wmx, wmx, grid_sz[0], device=device, dtype=torch.float32)
        .pow(2)
        .view(-1, 1, 1)
    )
    grid_y = (
        torch.linspace(-wmx, wmx, grid_sz[1], device=device, dtype=torch.float32)
        .pow(2)
        .view(1, -1, 1)
    )
    grid_z = (
        torch.linspace(-wmx, wmx, grid_sz[2], device=device, dtype=torch.float32)
        .pow(2)
        .view(1, 1, -1)
    )

    gp_tmp = torch.zeros(grid_sz, device=device, dtype=torch.float32)

    for i in range(l_scale.shape[0]):
        ker_x = torch.exp(-grid_x * l_scale[i, 0] ** 2)
        ker_y = torch.exp(-grid_y * l_scale[i, 1] ** 2)
        ker_z = torch.exp(-grid_z * l_scale[i, 2] ** 2)

        prod_l = torch.prod(l_scale[i])
        ker_3d = ker_x * ker_y * ker_z
        gp_tmp += prod_l * (ker_3d**2) * l_weights[i] ** 2

    gp_tmp = torch.sqrt(gp_tmp)

    noise = torch.randn(grid_sz, device=device, dtype=torch.float32) + 1j * torch.randn(
        grid_sz, device=device, dtype=torch.float32
    )
    gp_tmp = noise * gp_tmp

    gp_tmp = torch.fft.ifftshift(gp_tmp, dim=(0, 1, 2))
    gp_tmp = torch.fft.ifftn(gp_tmp, dim=(0, 1, 2))
    gp_tmp = torch.fft.ifftshift(gp_tmp, dim=(0, 1, 2))

    N_elem = gp_tmp.numel()
    gp_tmp = 2 * p_scale * bin_mask * np.sqrt(N_elem) * torch.real(gp_tmp) + mu

    return gp_tmp


def groupzproject(vol, chunk_size, mode="sum"):
    # vol: (H, W, Z)
    # chunk_size: int
    # project groups of Z
    Z = vol.shape[2]
    out_Z = int(np.ceil(Z / chunk_size))
    out = torch.zeros((vol.shape[0], vol.shape[1], out_Z), dtype=vol.dtype, device=vol.device)
    for i in range(out_Z):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, Z)
        if mode == "sum":
            out[:, :, i] = torch.sum(vol[:, :, start:end], dim=2)
        elif mode == "mean":
            out[:, :, i] = torch.mean(vol[:, :, start:end], dim=2)
    return out


@dataclass
class PsfParams:
    NA: float = 0.6
    objNA: float = 0.8
    n: float = 1.35
    lambda_exc: float = 0.92
    obj_fl: float = 4.5
    ss: int = 4
    sampling: int = 1
    zernikeDst: typing.Optional[np.ndarray] = None
    zernikeWt: typing.Optional[typing.Any] = None
    custom_phase: typing.Optional[np.ndarray] = None
    scatter_sz: typing.Optional[np.ndarray] = None
    scatter_wt: typing.Optional[np.ndarray] = None
    type: str = "gaussian"
    psf_sz: np.ndarray = field(
        default_factory=lambda: np.array([20.0, 20.0, 50.0], dtype=np.float32)
    )
    prop_sz: float = 10.0
    n_diff: float = 0.02
    fastmask: bool = True
    propcrop: bool = True
    taillength: typing.Optional[float] = 50.0
    bprof: str = "gaussian"
    length: typing.Optional[float] = None
    cylscale: typing.Optional[np.ndarray] = None
    scaling: typing.Optional[str] = None


def gaussian_beam_size(psf_params: PsfParams, dist: float, apod: float = 2.0) -> np.ndarray:
    radius = np.ceil(np.tan(np.arcsin(psf_params.objNA / psf_params.n)) * dist * 1.5) * apod
    return np.array([radius, radius, 0], dtype=np.float32)


def generate_gaussian_profile(
    X: torch.Tensor,
    Y: torch.Tensor,
    rad: float,
    aper: float,
    k: float,
    fl: float,
    offset=(0.0, 0.0),
):
    rho2 = X**2 + Y**2
    Uout = torch.exp(-((X - offset[0]) ** 2 + (Y - offset[1]) ** 2) / (rad**2))
    Uout = Uout * (rho2 < (aper**2))
    Uout = Uout * torch.exp(-1j * k / (2 * fl) * rho2)
    return Uout


def zrf(n, m, r):
    R = torch.zeros_like(r)
    for s in range(int((n - m) / 2) + 1):
        num = ((-1) ** s) * math.factorial(int(n - s))
        den = (
            math.factorial(int(s))
            * math.factorial(int((n + m) / 2 - s))
            * math.factorial(int((n - m) / 2 - s))
        )
        R += (num / den) * (r ** (n - 2 * s))
    return R


def zidx(i, j):
    idx = math.ceil(math.sqrt(0.25 + 2 * i) - 1.5)
    if j == 1:
        return idx
    elif j == 2:
        return ((idx + 1) % 2) * 2 * math.floor((i - (idx + 1) * idx / 2) / 2) + (idx % 2) * (
            2 * math.ceil((i - (idx + 1) * idx / 2) / 2) - 1
        )


def zernike(i, x, y):
    theta = torch.atan2(y, x)
    r = torch.sqrt(x**2 + y**2)
    n = zidx(i, 1)
    m = zidx(i, 2)

    if m == 0:
        Z = math.sqrt(n + 1) * zrf(n, 0, r)
    else:
        if i % 2 == 0:
            Z = math.sqrt(2 * (n + 1)) * zrf(n, m, r) * torch.cos(m * theta)
        else:
            Z = math.sqrt(2 * (n + 1)) * zrf(n, m, r) * torch.sin(m * theta)
    return Z


def generate_zernike(psf_params, offset=(0.0, 0.0)):
    if psf_params.zernikeWt is None:
        return 0
    if psf_params.zernikeDst is None:
        abb = np.array(psf_params.zernikeWt)
    else:
        abb = np.array(psf_params.zernikeWt)
    abb = abb * psf_params.lambda_exc * 1e-6
    return abb


def apply_zernike(Uin, X, Y, k, abb):
    if isinstance(abb, (int, float)) and abb == 0:
        return Uin
    phase = torch.zeros_like(X, dtype=torch.float32)
    idxs = np.nonzero(abb)[0]
    if len(idxs) > 0:
        for i in idxs:
            phase += abb[i] * zernike(i + 1, X, Y)
    phase = phase - torch.mean(phase)
    Uout = Uin * torch.exp(1j * k * phase)
    return Uout


def generate_bessel_profile(
    X, Y, rad, width, k, fl, bprof_type="gaussian", aper=float("inf"), offset=(0.0, 0.0)
):
    X2 = X - offset[0]
    Y2 = Y - offset[1]
    rho = torch.sqrt(X2**2 + Y2**2)
    if bprof_type == "uniform":
        Uout = ((rho - rad) < (width / 2)) * ((rho - rad) > -(width / 2))
        Uout = Uout.to(dtype=torch.float32)
    elif bprof_type == "posax":
        w = width / 1.133
        Uout = (
            (rho - rad + w / math.sqrt(2))
            * torch.exp(-(((rho - rad + w / math.sqrt(2)) / w) ** 2))
            * (rho - rad + w / math.sqrt(2) > 0)
        )
    elif bprof_type == "negax":
        w = width / 1.133
        Uout = (
            (rad - rho + w / math.sqrt(2))
            * torch.exp(-(((rad - rho + w / math.sqrt(2)) / w) ** 2))
            * (rad - rho + w / math.sqrt(2) > 0)
        )
    else:
        w = width / (2 * math.sqrt(math.log(2)))
        Uout = torch.exp(-(((rho - rad) / w) ** 2))
    rho2 = X**2 + Y**2
    Uout = Uout * (rho2 < (aper**2))
    Uout = Uout * torch.exp(-1j * k / (2 * fl) * rho2)
    return Uout


def generate_ba(
    vol_params: VolParams, psf_params: PsfParams, device=torch.device("cpu")
) -> torch.Tensor:
    if vol_params.vasc_sz is None:
        vol_params.vasc_sz = (
            gaussian_beam_size(psf_params, vol_params.vol_depth + vol_params.vol_sz[2] / 2)
            + vol_params.vol_sz
            + np.array([0, 0, 1]) * vol_params.vol_depth
        )

    fl = float(psf_params.obj_fl / 1000.0)
    D2 = float(1e-6 * (1.0 / vol_params.vres) / psf_params.ss)
    N = np.float32(1e-6 * (vol_params.vasc_sz[:2] - vol_params.vol_sz[:2]) / D2)

    beam_sz = gaussian_beam_size(psf_params, fl * 1e6)
    D1 = float(max(beam_sz) / 1e6 / min(N))
    nre = float(psf_params.n)
    rad = float(np.tan(np.arcsin(psf_params.NA / nre)) * fl)
    objrad = float(np.tan(np.arcsin(psf_params.objNA / nre)) * fl)
    k = 2 * nre * np.pi / float(psf_params.lambda_exc * 1e-6)

    x_range = torch.arange(-N[0] / 2, N[0] / 2, device=device) * D1
    y_range = torch.arange(-N[1] / 2, N[1] / 2, device=device) * D1
    Y, X = torch.meshgrid(y_range, x_range, indexing="ij")

    if psf_params.type == "bessel":
        length = psf_params.length if psf_params.length is not None else 60.0
        width = 1.4e-10 * np.tan(np.arcsin(psf_params.NA / nre)) / (fl * 1e-6 * 2 * length)
        bprof = psf_params.bprof if psf_params.bprof is not None else "gaussian"
        Uout = generate_bessel_profile(X, Y, rad, width, k, fl, bprof, aper=objrad)
    elif psf_params.type == "cylindrical":
        cylscale = psf_params.cylscale if psf_params.cylscale is not None else np.array([1.0, 1.0])
        Uout = generate_gaussian_profile(X / cylscale[0], Y / cylscale[1], rad, objrad, k, fl)
    else:
        Uout = generate_gaussian_profile(X, Y, rad, objrad, k, fl)

    if psf_params.custom_phase is not None:
        custom_phase = (
            torch.tensor(psf_params.custom_phase, dtype=torch.float32, device=device)
            .unsqueeze(0)
            .unsqueeze(0)
        )

        # [CRITICAL FIX: Coordinate Mapping]
        # In original NAOMi MATLAB, custom_phase was mapped to `objrad` (the objective NA=0.65 bounds).
        # However, the RayOptics/POPPY exported custom phase specifically represents the pupil
        # of the EXCITATION BEAM (NA=0.45), which corresponds to `rad`.
        # Mapping it to `objrad` stretched the aberration by ~1.55x, meaning the NA=0.45
        # beam only hit the central 69% of the phase canvas, missing the severe edge aberrations.
        # We now map the custom phase grid strictly to `rad` so the beam correctly experiences 100% of the intended aberration.
        grid_x = X / rad
        grid_y = Y / rad
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)
        phase_interp = F.grid_sample(
            custom_phase, grid, mode="nearest", padding_mode="zeros", align_corners=True
        ).squeeze()
        # custom_phase is expected to be in waves. Convert to radians:
        phase_interp = phase_interp * 2 * np.pi
        Uout = Uout * torch.exp(1j * phase_interp)
    elif psf_params.zernikeWt is not None:
        abb = generate_zernike(psf_params)
        if isinstance(abb, torch.Tensor) or np.any(abb != 0):
            Uout = apply_zernike(Uout, X / objrad, Y / objrad, k, abb)

    return Uout
