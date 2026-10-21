"""MaoMi: Modular anatomy and optics for Microscopy, building on NAOMi.

Public imports are lazy so the RANA core can be used without loading the
NAOMi-derived tissue, propagation or acquisition implementations.
"""
from importlib import import_module

__version__ = "0.8.0.dev0"
__author__ = "Yueze Wang"

_EXPORTS = {
    "zstack": "scanning", "OpticalConfig": "optics", "run_fixed": "fixed",
    "make_pupil": "pupils", "make_vtwins_pupils": "pupils", "simulate": "runtime",
    "FullFieldPSF": "rana", "CubicFullFieldRenderer": "rana",
    "RanaConfig": "rana", "RanaAddon": "rana", "RanaOperator": "rana",
    "DepthPSFBank": "rana", "DepthFullFieldRenderer": "rana",
    "VectorialOptics": "rana", "StratifiedMedium": "rana", "Acquisition": "rana",
    "PulseSpectrum": "rana", "EmissionResponse": "rana", "attach_rana": "rana",
    "Scene": "scene", "load_naomi_mat": "mat_io", "inspect_naomi_mat": "mat_io",
    "RunConfig": "workflow", "prepare_optics": "workflow", "render_static": "workflow",
    "render_scene": "workflow", "run_scene": "workflow", "save_scene": "archive",
    "load_scene": "archive", "save_optics": "archive", "load_optics": "archive",
    "MovieDataset": "dataset", "tissue_parameters": "tissue_config", "generate_scene": "tissue_config",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("." + _EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
