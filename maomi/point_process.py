"""Continuous exponential Hawkes process with O(K) recurrence state.

The full-history intensity is mu + sum_j A[:,mark_j] exp(-beta*(t-t_j)).
This is the linear, nonnegative continuous model, distinct from the rectified
inhibitory discrete model. No finite-history approximation or bin jitter.
"""

import numpy as np


def continuous_hawkes(
    mu, adjacency, duration, *, decay=1.0, rng=None, uniforms=None, max_events=1000000
):
    mu = np.asarray(mu, float).reshape(-1)
    a = np.asarray(adjacency, float)
    if (
        not len(mu)
        or a.shape != (len(mu), len(mu))
        or not np.isfinite(mu).all()
        or not np.isfinite(a).all()
        or np.any(mu < 0)
        or np.any(a < 0)
    ):
        raise ValueError("Finite nonnegative baseline and square excitation matrix required")
    if (
        not np.isfinite([duration, decay]).all()
        or duration < 0
        or decay <= 0
        or int(max_events) != max_events
        or max_events < 1
    ):
        raise ValueError("Invalid duration, decay or event budget")
    rng = np.random.default_rng() if rng is None else rng
    tape = None if uniforms is None else np.asarray(uniforms, float).ravel()
    cursor = 0

    def uniform():
        nonlocal cursor
        if tape is None:
            return float(rng.random())
        if cursor == len(tape):
            raise ValueError("Uniform tape exhausted")
        v = float(tape[cursor])
        cursor += 1
        if not 0 < v < 1:
            raise ValueError("Replay uniforms must be in (0,1)")
        return v

    excitation = np.zeros(len(mu))
    time = 0.0
    times = []
    marks = []
    proposals = 0
    complete = False
    while len(times) < max_events:
        upper = float(np.sum(mu + excitation))
        if upper == 0:
            complete = True
            break
        step = -np.log(max(uniform(), np.finfo(float).tiny)) / upper
        candidate = time + step
        if candidate > duration:
            complete = True
            break
        excitation *= np.exp(-decay * step)
        time = candidate
        proposals += 1
        rates = mu + excitation
        total = float(rates.sum())
        if uniform() < min(1.0, total / upper):
            target = uniform() * total
            mark = min(int(np.searchsorted(np.cumsum(rates), target, side="left")), len(mu) - 1)
            times.append(time)
            marks.append(mark)
            excitation += a[:, mark]
    if not complete and len(times) == max_events:
        raise RuntimeError(
            f"Event budget {max_events} reached at {time:.6g}s; requested duration is incomplete"
        )
    return dict(
        times=np.asarray(times),
        marks=np.asarray(marks, np.int64),
        duration=float(duration),
        complete=complete,
        proposals=proposals,
        uniforms_consumed=cursor if tape is not None else None,
    )


def bin_events(events, components, *, dt, frames):
    """Exact counts on half-open time bins; never collapse repeated events."""
    if (
        not np.isfinite([dt, frames, components]).all()
        or int(frames) != frames
        or int(components) != components
        or min(dt, frames, components) <= 0
    ):
        raise ValueError("Positive dimensions and dt required")
    times = np.asarray(events["times"], float)
    raw_marks = np.asarray(events["marks"])
    marks = raw_marks.astype(int)
    if (
        not np.isfinite(times).all()
        or np.any(raw_marks != marks)
        or times.shape != marks.shape
        or np.any(times < 0)
        or np.any(marks < 0)
        or np.any(marks >= components)
    ):
        raise ValueError("Invalid events")
    indices = np.floor(times / dt).astype(int)
    valid = indices < frames
    counts = np.zeros((int(components), int(frames)), np.int32)
    np.add.at(counts, (marks[valid], indices[valid]), 1)
    return counts
