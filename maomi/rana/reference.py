"""Explicit global cubic RBF reference renderer for optical experiments.

The common acquisition workflow uses RanaOperator's compact positive cubic
weights. This reference keeps the historical comparison's numerical operator.
"""
import time
import numpy as np


def render_reference(volume, *, config=None, psf_field=None, voxel_size_um=None,
                     origin_um=(0., 0.), device=None, status_window=True, z_origin_um=0.):
    """Render full kernels with global cubic RBF XY and linear depth weights."""
    from .fields import FullFieldPSF, CubicFullFieldRenderer
    from ..rana_ui import RanaStatus
    from .depth import DepthPSFBank, DepthFullFieldRenderer
    from .config import RanaConfig
    if config is not None:
        if not isinstance(config, RanaConfig) or psf_field is not None:
            raise TypeError("Use a RanaConfig or a complete custom PSF field")
        if voxel_size_um is None:
            voxel_size_um = config.voxel_size_um
    elif not isinstance(psf_field, (FullFieldPSF, DepthPSFBank)):
        raise TypeError("RANA requires a complete FullFieldPSF or DepthPSFBank")
    if not isinstance(psf_field, DepthPSFBank) and config is None and z_origin_um != 0:
        raise ValueError("z_origin_um applies to depth banks")
    if voxel_size_um is None:
        raise ValueError("Specify the image voxel_size_um explicitly")
    volume = np.asarray(volume)
    selected_device = device or "cpu"
    started = time.perf_counter()
    with RanaStatus(status_window) as status:
        prepared_at = time.perf_counter()
        if config is not None:
            psf_field = config.build_field()
        optics_at = time.perf_counter()
        depth_bank = isinstance(psf_field, DepthPSFBank)
        renderer_class = DepthFullFieldRenderer if depth_bank else CubicFullFieldRenderer
        renderer = renderer_class(psf_field, volume.shape, voxel_size_um=voxel_size_um,
                                  origin_um=origin_um, device=selected_device,
                                  **({'z_origin_um': z_origin_um} if depth_bank else {}))
        rendered_at = time.perf_counter()
        image = renderer(volume)
        finished_at = time.perf_counter()
    return {"image": image, "mode": "rana", "operator": psf_field.operator,
            "timing_seconds": {"status_start": prepared_at-started,
                               "optics": optics_at-prepared_at,
                               "preparation": rendered_at-optics_at,
                               "render": finished_at-rendered_at},
            "simulation_seconds": finished_at-prepared_at,
            "total_seconds": time.perf_counter()-started,
            "status_window": status.status,
            "validation": renderer.validation,
            "settings": {"mode": "rana", "device": str(selected_device),
                         "field_nodes": list((psf_field.fields[0] if depth_bank else psf_field).kernels.shape[:2]),
                         "depth_nodes_um": psf_field.depths_um.tolist() if depth_bank else None,
                         "z_origin_um": float(z_origin_um),
                         "psf_provenance": ([f.provenance for f in psf_field.fields] if depth_bank
                                            else psf_field.provenance),
                         "voxel_size_um": list(voxel_size_um), "origin_um": list(origin_um),
                         "interpolation": "full-kernel affine-augmented cubic RBF",
                         "shared_psf_fallback": False}}
