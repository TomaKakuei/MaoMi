"""MaoMi: Modular anatomy and optics for Microscopy, building on NAOMi."""

from .scanning import zstack
from .optics import OpticalConfig

__version__ = "0.8.0.dev0"
__author__ = "Yueze Wang"


def run_fixed(*args, **kwargs):
    """Lazy entry point; CPU reference imports do not require CuPy."""
    from .fixed import run_fixed as implementation

    return implementation(*args, **kwargs)


from .pupils import make_pupil, make_vtwins_pupils
from .enhanced import FullFieldPSF, CubicFullFieldRenderer
from .runtime import simulate
from .rana import RanaConfig
from .rana_depth import DepthPSFBank, DepthFullFieldRenderer
from .rana_optics import VectorialOptics, StratifiedMedium, Acquisition, PulseSpectrum, EmissionResponse
from .rana_addon import RanaAddon, attach_rana
from .scene import Scene
from .mat_io import load_naomi_mat, inspect_naomi_mat
from .workflow import RunConfig, prepare_optics, render_static, run_scene
from .archive import save_scene, load_scene, save_optics, load_optics
from .dataset import MovieDataset
from .tissue_config import tissue_parameters, generate_scene
