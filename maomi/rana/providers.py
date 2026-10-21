"""PSF preparation boundary shared by independent optics providers and hosts."""
from typing import Protocol, TypeAlias, runtime_checkable

from .fields import FullFieldPSF
from .depth import DepthPSFBank

ResolvedField: TypeAlias = FullFieldPSF | DepthPSFBank


@runtime_checkable
class PSFProvider(Protocol):
    """Build sampled effective-intensity fields in specimen micrometres.

    Providers own their solver, calibration and sampling configuration. They
    return numerical fields with provenance; rendering and archives need no
    access to the provider or its dependencies. No inheritance or registration
    is required. RanaConfig implements this protocol.
    """

    def build_field(self) -> ResolvedField: ...


FieldSource: TypeAlias = ResolvedField | PSFProvider


def resolve_field(source: FieldSource) -> ResolvedField:
    """Resolve once during preparation, never within the frame loop.

    Returned fields enforce intensity, units, grid and provenance contracts.
    The acquisition operator additionally requires XYZ kernels and coverage.
    """
    if isinstance(source, (FullFieldPSF, DepthPSFBank)):
        return source
    build = getattr(source, 'build_field', None)
    if not callable(build):
        raise TypeError('Supply FullFieldPSF, DepthPSFBank or a PSFProvider with build_field()')
    field = build()
    if not isinstance(field, (FullFieldPSF, DepthPSFBank)):
        raise TypeError('PSFProvider.build_field() must return FullFieldPSF or DepthPSFBank')
    return field
