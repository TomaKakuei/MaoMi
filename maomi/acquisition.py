"""Exposure clocks and streaming integration, independent of any optical solver.

Arrays are XY. The slow scan/row axis is Y; X is the fast raster axis.
Render callbacks return instantaneous expected photon rates, never noisy images.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AcquisitionClock:
    mode: str = "global"
    exposure_s: float = .001
    line_period_s: float = 0.
    pixel_period_s: float = 0.
    reverse: bool = False
    bidirectional: bool = False
    samples: int = 65

    def __post_init__(self):
        if self.mode not in ("global", "rolling", "raster"):
            raise ValueError("Clock mode must be global, rolling or raster")
        if not np.isfinite([self.exposure_s, self.line_period_s, self.pixel_period_s]).all():
            raise ValueError("Clock durations must be finite seconds")
        if self.exposure_s <= 0 or min(self.line_period_s, self.pixel_period_s) < 0:
            raise ValueError("Exposure must be positive; clock periods must be nonnegative")
        if not isinstance(self.samples, (int, np.integer)) or isinstance(self.samples, bool) or self.samples < 2:
            raise ValueError("samples must be an integer >= 2")
        if self.mode == "global" and self.line_period_s != 0:
            raise ValueError("Global exposure has no line period")
        if self.mode != "raster" and (self.pixel_period_s != 0 or self.bidirectional):
            raise ValueError("Pixel timing and bidirectional scans require raster mode")
        if self.mode == "raster" and self.pixel_period_s < self.exposure_s:
            raise ValueError("Raster pixel period must include the pixel dwell time")

    def offsets(self, shape):
        """Pixel exposure midpoints relative to the common frame midpoint."""
        if len(shape) != 2 or any(int(n) != n or n < 1 for n in shape):
            raise ValueError("Supply a positive XY image shape")
        nx, ny = map(int, shape)
        if self.mode == "raster" and self.line_period_s < nx*self.pixel_period_s:
            raise ValueError("Raster line period must include all pixels and any flyback")
        rows = np.arange(ny)
        rank = rows[::-1] if self.reverse else rows
        offsets = np.broadcast_to(rank[None, :]*self.line_period_s, (nx, ny)).copy()
        if self.mode == "raster":
            x = np.broadcast_to(np.arange(nx)[:, None], (nx, ny))
            x = np.where((rank[None, :] % 2 == 1) & self.bidirectional, nx-1-x, x)
            offsets += x*self.pixel_period_s
        return offsets - (offsets.min()+offsets.max())/2

    def plan(self, shape):
        return ExposurePlan(self, shape)


class ExposurePlan:
    """Exact box integration of a piecewise-linear sequence of rate snapshots.

    Each snapshot is rendered once, then discarded. Memory is O(pixels), not
    O(samples * pixels). Increase clock.samples to refine changing scenes.
    """
    def __init__(self, clock, shape):
        self.clock = clock
        self.offsets = clock.offsets(shape)
        self.shape = self.offsets.shape
        self.lower = self.offsets-clock.exposure_s/2
        self.upper = self.offsets+clock.exposure_s/2
        self.nodes = np.linspace(self.lower.min(), self.upper.max(), clock.samples)
        self.step = self.nodes[1]-self.nodes[0]

    def weights(self, node):
        """Integrated linear hat function, in seconds, for one snapshot."""
        h = self.step
        a = np.clip((self.lower-(node-h))/h, 0., 1.)
        b = np.clip((self.upper-(node-h))/h, 0., 1.)
        rising = (b-a)*(b+a)/2
        a = np.clip((self.lower-node)/h, 0., 1.)
        b = np.clip((self.upper-node)/h, 0., 1.)
        return h*(rising + (b-a)*(1-(b+a)/2))

    def integrate(self, render_rate, *, center_s=0.):
        if not np.isfinite(center_s):
            raise ValueError("Frame midpoint must be finite")
        result = np.zeros(self.shape, dtype=np.float64)
        for node in self.nodes:
            weight = self.weights(node)
            if not np.any(weight):
                continue
            rate = np.asarray(render_rate(float(center_s+node)))
            if (rate.shape != self.shape or np.iscomplexobj(rate)
                    or not np.isfinite(rate).all() or (rate < 0).any()):
                raise ValueError("Renderer must return finite nonnegative XY photon rates")
            result += rate*weight
        return result
