"""Excitation power and the source-scoped temporal-focusing axial restriction."""

import numpy as np

ORDERS = {"one-photon": 1, "two-photon": 2, "three-photon": 3, "temporal-focusing": 2}


def excitation_order(config):
    if config.scaling not in ORDERS:
        raise ValueError(f"Unknown excitation scaling: {config.scaling}")
    return ORDERS[config.scaling]


def temporal_profile(size, length_um, dz_um, *, legacy_offset=0.0):
    """Exact applyTemporalFocusing convention; offset is the legacy argument.

    Its source uses offset*dz, not a physical-micron offset divided by dz.
    Keep that argument explicitly named rather than silently changing units.
    """
    if (
        int(size) != size
        or size < 1
        or not np.isfinite([length_um, dz_um, legacy_offset]).all()
        or min(length_um, dz_um) <= 0
    ):
        raise ValueError("Positive finite axial length and sampling required")
    rayleigh = length_um / (dz_um * 2 * np.sqrt(3))
    indices = np.ceil(size / 2) - np.arange(1, size + 1) + legacy_offset * dz_um
    return 1 / np.sqrt(1 + (indices / rayleigh) ** 2)


def axial_profile(config, size, part="focus"):
    if config.scaling != "temporal-focusing":
        return np.ones(size, np.float32)
    if part == "focus":
        dz = 1 / config.vres
        offset = 0.0
    else:
        pp = config.prop_um * config.vres
        depth = config.psf_um[2] * config.vres
        dz = pp / config.vres
        offset = (
            (depth * config.vres + 2 * pp) / (pp * pp)
            if part == "top"
            else -depth * config.vres / (pp * pp)
        )
    return temporal_profile(size, config.temporal_length_um, dz, legacy_offset=offset).astype(
        np.float32
    )
