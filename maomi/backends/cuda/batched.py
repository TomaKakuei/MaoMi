"""Persistent geometry, batched cuFFT, CUDA C gathers and fused PSF resampling.

Constant-pitch Fresnel stages commute with the input/output shifts. Moving Q
to FFT order once removes four shift copies per screen, without approximation.
"""

import numpy as np
import copy
import torch
import torch.nn.functional as F
from maomi.optics import pupil
from maomi._propagation import geometry, edge_apodization
from maomi.numerics import fresnel, resize_xy
from .native import gather, ResamplePlan
from maomi.excitation import excitation_order, axial_profile


class OpticalPlan:
    def __init__(self, config, raw_fields, *, uin=None, device="cuda", native=True):
        if not str(device).startswith("cuda"):
            raise ValueError("Use reference optics for CPU")
        self.config = copy.deepcopy(config)
        self.device = device
        self.native = native
        self.order = excitation_order(config)
        r, ss = config.vres, config.ss
        g = geometry(config, raw_fields)
        self.n = n = g.n
        self.n2 = n2 = g.n2
        self.c = g.crop
        self.fc = g.focal_crop
        fshape = g.focal_shape
        d1, d2, fl = g.d1, g.d2, g.fl
        self.wvl = wvl = g.wvl
        self.nr = nr = g.nr
        u = pupil(config, device=device) if uin is None else torch.as_tensor(uin, device=device)
        if u.dtype != torch.complex64:
            raise ValueError(
                "This native backend requires complex64; use reference optics for other precision."
            )
        self.sg = edge_apodization(n, device, u.real.dtype)
        self.sg2 = edge_apodization(n2, device, u.real.dtype)
        base = fresnel(
            u,
            wvl,
            [d1, d2],
            [0, np.float32(fl - (config.depth_um + config.volume_um[2] / 2) * 1e-6)],
            torch.stack((self.sg, self.sg), -1),
            nr,
        )
        self.base = base / torch.linalg.vector_norm(base)
        scale = config.n_diff * 2 * np.pi / (config.wavelength_um * r)
        self.phases = [
            torch.exp(
                1j * resize_xy(torch.as_tensor(a, device=device, dtype=u.real.dtype), ss) * scale
            )
            .permute(2, 0, 1)
            .contiguous()
            for a in raw_fields
        ]
        a, b, c = [x.shape[-1] for x in raw_fields]
        self.z_positions = g.z
        self.cut = g.cut
        self.axial = {
            key: torch.as_tensor(axial_profile(config, size, key), device=device)
            for key, size in [("focus", b), ("top", a + 1 - self.cut), ("bottom", c + 1)]
        }
        self.q = []
        for size, z in zip((n, n2, n), self.z_positions):
            x = torch.arange(-size / 2, size / 2, device=device)[:, None]
            y = torch.arange(-size / 2, size / 2, device=device)[None, :]
            kernels = []
            for dz in np.diff(z):
                q = torch.exp(
                    -1j
                    * np.pi
                    * (float(wvl) / float(nr))
                    * dz
                    * ((x / (size * d2)) ** 2 + (y / (size * d2)) ** 2)
                )
                kernels.append(torch.fft.ifftshift(q))
            self.q.append(kernels)
        self.focal_resample = ResamplePlan(*fshape, ss, device)
        self.tail_resample = ResamplePlan(n2, n2, ss, device)
        self.scale = ss**4 * (r * (1e6 * wvl) ** 1.5) / (np.pi * nr)
        self.cache = {}
        self.paths_evaluated = 0
        self.graphs = {}

    def _focal_gpu(self, off):
        ua, _ = self._stage(
            self.base[None].expand(len(off), -1, -1), self._screens(0, off), 0, save=False
        )
        _, focus = self._stage(ua[:, self.c, self.c], self._screens(1, off), 1, crop=self.fc)
        return self.focal_resample(
            focus[:, :-1],
            self.scale,
            native=self.native,
            order=self.order,
            axial=self.axial["focus"],
        )

    def prepare_graph(self, batch_size=32):
        """Capture a fixed-size focal batch; later calls update only coordinates."""
        if not self.native:
            raise ValueError("Graph requires native gather without CPU indexing")
        if batch_size in self.graphs:
            return
        off = torch.zeros((batch_size, 2), device=self.device, dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                output = self._focal_gpu(off)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.inference_mode():
            output = self._focal_gpu(off)
        self.graphs[batch_size] = (graph, off, output)

    @torch.inference_mode()
    def evaluate_graph(self, points_um, *, batch_size=32):
        self.prepare_graph(batch_size)
        pts = np.asarray(points_um, float).reshape(-1, 2)
        scaled = pts * self.config.vres * self.config.ss
        if np.any(pts < 0) or np.any(pts > np.asarray(self.config.volume_um[:2])):
            raise ValueError("Outside FOV")
        if not np.allclose(scaled, np.round(scaled)):
            raise ValueError("Points must lie on supersampled grid")
        offsets = np.round(scaled).astype(np.int32)
        graph, buf, out = self.graphs[batch_size]
        collected = []
        for s in range(0, len(pts), batch_size):
            part = offsets[s : s + batch_size]
            padded = np.zeros((batch_size, 2), np.int32)
            padded[: len(part)] = part
            buf.copy_(torch.from_numpy(padded))
            graph.replay()
            collected.append(out[: len(part)].cpu().numpy().copy())
            self.paths_evaluated += len(part)
        return {
            "psfs": np.concatenate(collected),
            "points_um": pts,
            "tails_computed": False,
            "graph_padding_slots": (int(np.ceil(len(pts) / batch_size)) * batch_size - len(pts)),
        }

    def _screens(self, part, offsets):
        n = self.n2 if part == 1 else self.n
        edge = self.sg2 if part == 1 else self.sg
        if self.native:
            return gather(self.phases[part], edge, offsets, n)
        # Python/torch control for native ablation, same tensor layouts.
        pairs = offsets.cpu().numpy()
        return torch.stack([self.phases[part][:, x : x + n, y : y + n] * edge for x, y in pairs])

    def _stage(self, u, screens, part, crop=None, start=0, save=True):
        u = u * screens[:, 0]
        saved = []
        if save and start == 0:
            saved.append(u[..., crop[0], crop[1]].clone() if crop else u.clone())
        for i, q in enumerate(self.q[part]):
            u = torch.fft.ifft2(torch.fft.fft2(u) * q)
            if i + 1 < screens.shape[1]:
                u = u * screens[:, i + 1]
            if save and i + 1 >= start:
                saved.append(u[..., crop[0], crop[1]].clone() if crop else u.clone())
        return u, torch.stack(saved, dim=1) if save else None

    @torch.inference_mode()
    def evaluate(self, points_um, *, batch_size=16, tails=False):
        pts = np.asarray(points_um, float).reshape(-1, 2)
        offsets = pts * self.config.vres * self.config.ss
        if np.any(pts < 0) or np.any(pts > np.asarray(self.config.volume_um[:2])):
            raise ValueError("Outside FOV")
        if not np.allclose(offsets, np.round(offsets)):
            raise ValueError("Points must lie on supersampled grid")
        collected = []
        tail_out = [[], []]
        for s in range(0, len(pts), batch_size):
            bpts = pts[s : s + batch_size]
            off = torch.tensor(
                np.round(offsets[s : s + batch_size]).astype(np.int32), device=self.device
            )
            ua, top = self._stage(
                self.base[None].expand(len(bpts), -1, -1),
                self._screens(0, off),
                0,
                crop=(self.c, self.c),
                start=self.cut,
                save=tails,
            )
            ub, focus = self._stage(ua[:, self.c, self.c], self._screens(1, off), 1, crop=self.fc)
            psfs = self.focal_resample(
                focus[:, :-1],
                self.scale,
                native=self.native,
                order=self.order,
                axial=self.axial["focus"],
            )
            collected.append(psfs.cpu().numpy())
            if tails:
                _, bottom = self._stage(
                    F.pad(ub, ((self.n - self.n2) // 2,) * 4),
                    self._screens(2, off),
                    2,
                    crop=(self.c, self.c),
                )
                for j, field in enumerate((top, bottom)):
                    axial = self.axial["top" if j == 0 else "bottom"]
                    intensity = (
                        self.config.ss**2
                        * field.abs() ** (2 * self.order)
                        * torch.as_tensor(axial, device=self.device)[None, :, None, None]
                    )
                    profile = intensity.sum((-2, -1))
                    trapezoid = torch.ones(field.shape[1], device=self.device)
                    trapezoid[[0, -1]] = 0.5
                    weight = (profile * trapezoid).sum(-1) * self.config.prop_um
                    images = self.tail_resample(
                        field, self.config.ss**4, native=self.native, order=self.order, axial=axial
                    )
                    convmask = (images * trapezoid).sum(-1)
                    tail_out[j].append(
                        (convmask.cpu().numpy(), weight.cpu().numpy(), profile.cpu().numpy())
                    )
            self.paths_evaluated += len(bpts)
        out = {"psfs": np.concatenate(collected), "points_um": pts, "tails_computed": tails}
        for j, name in enumerate(("top", "bottom")):
            if tails:
                out[name] = {
                    k: np.concatenate([item[i] for item in tail_out[j]])
                    for i, k in enumerate(("convmask", "weight", "profile"))
                }
        return out

    def cached(self, points_um, *, batch_size=16):
        pts = np.asarray(points_um, float).reshape(-1, 2)
        if np.any(pts < 0) or np.any(pts > np.asarray(self.config.volume_um[:2])):
            raise ValueError("Outside FOV")
        scaled = pts * self.config.vres * self.config.ss
        if not np.allclose(scaled, np.round(scaled)):
            raise ValueError("Points must lie on supersampled grid")
        keys = [tuple(np.round(p * self.config.vres * self.config.ss).astype(int)) for p in pts]
        missing = list(dict.fromkeys(k for k in keys if k not in self.cache))
        if missing:
            pos = np.array(missing) / (self.config.vres * self.config.ss)
            out = self.evaluate(pos, batch_size=batch_size)
            self.cache.update(zip(missing, out["psfs"]))
        return np.stack([self.cache[k] for k in keys])
