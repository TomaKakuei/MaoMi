"""MAOMi: fixed-grid optical propagation and static imaging derived from NAOMi."""

from .scanning import zstack
from .optics import OpticalConfig

__version__ = "0.5.0.dev0"


def run_fixed(*args, **kwargs):
    """Lazy entry point; CPU reference imports do not require CuPy."""
    from .fixed import run_fixed as implementation

    return implementation(*args, **kwargs)


from .pupils import make_pupil, make_vtwins_pupils
