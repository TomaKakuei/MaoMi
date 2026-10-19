"""MaoMi: Modular anatomy and optics for Microscopy, building on NAOMi."""

from .scanning import zstack
from .optics import OpticalConfig

__version__ = "0.7.0.dev0"


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
