"""Compatibility API for older notebooks; use the scene workflow for new work."""
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
    import warnings
    from .rana.reference import render_reference
    warnings.warn("simulate(mode='rana') is deprecated; use run_scene/render_scene with RANA, "
                  "or maomi.rana.reference.render_reference for the original global RBF operator.",
                  DeprecationWarning, stacklevel=2)
    return render_reference(volume, config=config, psf_field=psf_field, voxel_size_um=voxel_size_um,
                            origin_um=origin_um, device=device, status_window=status_window,
                            z_origin_um=z_origin_um)
