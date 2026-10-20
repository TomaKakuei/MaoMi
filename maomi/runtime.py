"""Explicit standard / RANA Mode selection for scripts, notebooks and the CLI."""
import time
import numpy as np


def simulate(volume, *, mode="standard", config=None, psf_field=None,
             voxel_size_um=None, origin_um=(0., 0.), device=None,
             status_window=True, z_origin_um=0., **native_options):
    """Run MATLAB-compatible MaoMi or theoretically higher-accuracy RANA Mode.

    Standard mode uses a native OpticalConfig. RANA accepts RanaConfig,
    FullFieldPSF or DepthPSFBank and retains local shape and gain. Its optional
    desktop cat is isolated from the numerical operator.
    """
    if mode not in {"standard", "rana"}:
        raise ValueError("mode must be 'standard' or 'rana'")
    if mode == "standard":
        if config is None:
            raise ValueError("Standard mode requires config=OpticalConfig(...)")
        if psf_field is not None or voxel_size_um is not None or tuple(origin_um) != (0., 0.) or z_origin_um != 0:
            raise ValueError("Custom full-field inputs require mode='rana'")
        if device in ("cpu","auto"):
            from .workflow import prepare_optics,select_device,render_static
            selected = select_device(device)
            result = prepare_optics(config,device=selected,**native_options)
            started = time.perf_counter()
            result["image"] = render_static(volume,result,device=selected)
            elapsed = time.perf_counter()-started
            result.setdefault("timing_seconds",{})["static_scan"] = elapsed
            result["total_seconds"] += elapsed
            result["mode"] = "standard"
            return result
        from .fixed import run_fixed
        result = run_fixed(config, volume, device=device or "cuda", **native_options)
        result["mode"] = "standard"
        return result
    if native_options:
        raise ValueError("Native propagation options belong to standard mode")
    from .enhanced import FullFieldPSF, CubicFullFieldRenderer
    from .rana_ui import RanaStatus
    from .rana_depth import DepthPSFBank, DepthFullFieldRenderer
    from .rana import RanaConfig
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
