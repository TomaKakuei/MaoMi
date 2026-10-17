"""Small portable synthetic inputs; no anatomy or calcium-generation claim."""

import numpy as np
from .optics import OpticalConfig
from .fixed import field_shapes


def synthetic_scene(scene="scattering", seed=20260927):
    if scene not in ("uniform", "scattering", "vessel"):
        raise ValueError("Unknown demo scene")
    cfg = OpticalConfig(
        volume_um=(16.0, 16.0, 10.0), depth_um=20.0, vres=2.0, tail_um=10.0, seed=seed
    )
    x, y, z = np.meshgrid(np.arange(32), np.arange(32), np.arange(20), indexing="ij")
    volume = (
        np.exp(-((x - 11) ** 2 + (y - 15) ** 2 + (z - 9) ** 2) / 9)
        + 0.7 * np.exp(-((x - 23) ** 2 + (y - 21) ** 2 + (z - 12) ** 2) / 8)
    ).astype(np.float32)
    vessels = None
    if scene == "vessel":
        shapes = field_shapes(cfg)
        depth = round(
            (cfg.depth_um + cfg.volume_um[2] / 2 + cfg.psf_um[2] / 2 + cfg.tail_um) * cfg.vres
        )
        vessels = np.zeros((*shapes[0][:2], depth), np.float32)
        xx, yy = np.meshgrid(
            np.arange(vessels.shape[0]), np.arange(vessels.shape[1]), indexing="ij"
        )
        for iz in range(depth):
            vessels[:, :, iz] = (xx - (vessels.shape[0] / 2 - 14 + 0.22 * iz)) ** 2 + (
                yy - vessels.shape[1] / 2
            ) ** 2 < 64
    return cfg, volume, vessels
