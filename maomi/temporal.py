"""Causal activity models with explicit clocks and component axes.

These interfaces repair ambiguous legacy wrappers rather than reproduce their
axis/buffering bugs. Units and model equations are explicit in each entry point.
"""

from fractions import Fraction
import numpy as np
from scipy.signal import lfilter, resample_poly
from .activity import calcium_dynamics


def _matrix(value, name):
    a = np.asarray(value, float)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite components-by-time matrix")
    return a


class ARTraceFilter:
    """AR1 decay or AR2 rise/decay; causal along time, independent per row.

    tau values and dt are seconds. A unit impulse has peak `amplitude` above
    `baseline`. AR2's zero-lag response is zero. Filter state persists across
    process calls; splitting a trace does not alter its output.
    """

    def __init__(
        self, components, *, dt, tau_decay=0.4, tau_rise=None, baseline=1.0, amplitude=1.0
    ):
        if int(components) != components or components < 1:
            raise ValueError("components must be positive integer")
        if not np.isfinite([dt, tau_decay]).all() or min(dt, tau_decay) <= 0:
            raise ValueError("Positive finite time constants required")
        self.components = int(components)
        p = float(np.exp(-dt / tau_decay))
        if tau_rise is None:
            self.b = np.array([1.0])
            self.a = np.array([1.0, -p])
        else:
            if not np.isfinite(tau_rise) or not 0 < tau_rise < tau_decay:
                raise ValueError("AR2 requires 0 < tau_rise < tau_decay")
            q = float(np.exp(-dt / tau_rise))
            peak_at = np.log(np.log(q) / np.log(p)) / np.log(p / q)
            k = np.array([max(1, int(np.floor(peak_at))), max(1, int(np.ceil(peak_at)))])
            peak = float(np.max(p**k - q**k))
            self.b = np.array([0.0, (p - q) / peak])
            self.a = np.array([1.0, -(p + q), p * q])
        self.state = np.zeros((self.components, max(len(self.a), len(self.b)) - 1))
        self.baseline = np.broadcast_to(np.asarray(baseline, float), (self.components,)).copy()
        self.amplitude = np.broadcast_to(np.asarray(amplitude, float), (self.components,)).copy()
        if not np.isfinite(self.baseline).all() or not np.isfinite(self.amplitude).all():
            raise ValueError("Finite fluorescence scales required")

    def process(self, spike_counts):
        counts = _matrix(spike_counts, "spike_counts")
        if counts.shape[0] != self.components or np.any(counts < 0):
            raise ValueError("Nonnegative counts, one row per component required")
        if counts.shape[1] == 0:
            return np.empty_like(counts)
        out, self.state = lfilter(self.b, self.a, counts, axis=1, zi=self.state)
        return self.baseline[:, None] + self.amplitude[:, None] * out


def voltage_fluorescence(
    event_times, *, dt=0.00125, frames, tau=0.02, amplitude=0.2, baseline=1.0, delay=0.0
):
    """Clean fast fluorescence from event times in seconds, not membrane mV.

    Matches the upstream spk_calcium 1exp default linear response, called with
    explicit event-time cells and acquisition clock; no calcium-unit rescaling.
    Noise belongs to the detector stage, avoiding a second hidden noise source.
    """
    if (
        int(frames) != frames
        or frames < 1
        or not np.isfinite([dt, tau, delay]).all()
        or min(dt, tau) <= 0
    ):
        raise ValueError("Invalid sampling parameters")
    if not np.isfinite(baseline).all() or not np.isfinite(amplitude).all():
        raise ValueError("Finite fluorescence scales required")
    frames = int(frames)
    out = np.empty((len(event_times), frames))
    baseline = np.broadcast_to(baseline, (len(event_times),))
    amplitude = np.broadcast_to(amplitude, (len(event_times),))
    for j, events in enumerate(event_times):
        times = np.asarray(events, float).reshape(-1) + delay
        if not np.isfinite(times).all() or np.any(times < 0):
            raise ValueError("Shifted event times must be finite and nonnegative")
        bins = np.ceil(times / dt).astype(int)
        valid = (bins >= 0) & (bins < frames)
        increase = np.zeros(frames)
        np.add.at(increase, bins[valid], np.exp(-(bins[valid] * dt - times[valid]) / tau))
        response = lfilter([1.0], [1.0, -np.exp(-dt / tau)], increase)
        out[j] = baseline[j] * (1 + amplitude[j] * response)
    return out


def resample_fluorescence(traces, *, input_dt, output_dt):
    """Rational-clock antialiased resampling; baseline floor retained."""
    a = _matrix(traces, "traces")
    if min(input_dt, output_dt) <= 0 or not np.isfinite([input_dt, output_dt]).all():
        raise ValueError("Positive sample clocks required")
    if input_dt == output_dt:
        return a.copy()
    ratio = Fraction(float(input_dt / output_dt)).limit_denominator(10000)
    if abs(float(ratio) - input_dt / output_dt) > 1e-10:
        raise ValueError("Sampling ratio cannot be represented accurately")
    if not a.shape[1]:
        raise ValueError("At least one input sample required")
    # A denominator-multiple pad preserves the t=0 output clock exactly.
    pad = max(1, int(np.ceil((1 / input_dt) / ratio.denominator))) * ratio.denominator
    trim = pad * ratio.numerator // ratio.denominator
    n = int(np.floor(a.shape[1] * input_dt / output_dt))
    padded = np.pad(a, ((0, 0), (pad, pad)), mode="edge")
    result = resample_poly(padded, ratio.numerator, ratio.denominator, axis=1)[:, trim : trim + n]
    return np.maximum(result, a.min(axis=1, keepdims=True))


def background_from_spikes(spike_counts, *, output_dt=1 / 30, protein="GCaMP6f", modulation=1.0):
    """Independent neuropil components, with explicit 100 Hz count inputs.

    Uses original background Ca_DE equations (extrusion 2800, 500-sample warmup).
    Background modulation has its own component count. It never reads S_cell or
    reuses soma modulation. The optional old truncation-dependent 0.4 scaling
    is replaced by the caller's explicit modulation.
    """
    counts = _matrix(spike_counts, "background spike_counts")
    if np.any(counts < 0):
        raise ValueError("Spike counts cannot be negative")
    buffered = np.pad(counts * 7.6e-6, ((0, 0), (500, 0)))
    _, _, trace = calcium_dynamics(
        buffered, {"sat_type": "Ca_DE", "dt": 0.01, "ext_rate": 2800}, protein
    )
    trace = resample_fluorescence(trace[:, 500:], input_dt=0.01, output_dt=output_dt)
    mod = np.broadcast_to(np.asarray(modulation, float), (len(trace),))
    if not np.isfinite(mod).all():
        raise ValueError("Finite modulation required")
    return trace * mod[:, None]
