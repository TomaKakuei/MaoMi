"""Small API examples; choose fresh export paths before running the functions."""

import numpy as np
from maomi.temporal import ARTraceFilter, voltage_fluorescence, background_from_spikes
from maomi.point_process import continuous_hawkes, bin_events
from maomi.movie import iter_scan_volume
from maomi.movie_io import write_movie
from maomi.analysis import ideal_components, detector_snr
from maomi.pupils import make_pupil
from maomi import run_fixed


def temporal_example():
    # Seconds throughout; spike counts are not calcium concentrations.
    events = continuous_hawkes(
        [0.5, 0.8], [[0.1, 0.02], [0.03, 0.1]], 10.0, decay=2.0, rng=np.random.default_rng(5)
    )
    counts = bin_events(events, 2, dt=1 / 30, frames=300)
    filt = ARTraceFilter(2, dt=1 / 30, tau_decay=0.4, tau_rise=0.06)
    traces = np.concatenate([filt.process(counts[:, :80]), filt.process(counts[:, 80:])], axis=1)
    voltage = voltage_fluorescence([[0.01, 0.1], [0.05]], dt=0.00125, frames=160)
    background = background_from_spikes(np.zeros((3, 1000)), output_dt=1 / 30)
    return traces, voltage, background


def export_example(neurons, optics, activity, scan, path, shape, frames):
    # All activity inputs are retained; generated movie frames are streamed.
    stream = iter_scan_volume(neurons, optics, activity, scan, add_noise=False)
    return write_movie(stream, path, shape=shape, frame_count=frames)
    # For AVI additionally supply intensity_range=(low,high); clipping is rejected.


def pupil_example(config, volume, vessels):
    field = make_pupil(config, kind="bessel", profile="gaussian", length_um=60.0)
    return run_fixed(config, volume, vessels=vessels, input_pupil=field)


def analysis_example(neurons, optics, activity, scan):
    result = ideal_components(neurons, optics, activity, scan, grouping="by_compartment")
    return result, detector_snr(
        np.maximum(result["components"], 0), np.maximum(result["baseline"], 0)[:, :, None]
    )
