"""RANA: Ray Aware Nonuniform Acquisition.

Independent optical fields, physics and rendering; host coupling is in adapters.
"""
from .config import RanaConfig, RanaAddon
from .fields import FullFieldPSF, CubicFullFieldRenderer
from .depth import DepthPSFBank, DepthFullFieldRenderer
from .physics import VectorialOptics, StratifiedMedium, Acquisition, PulseSpectrum, EmissionResponse
from .operator import RanaOperator
from .adapters.naomi import attach_rana
