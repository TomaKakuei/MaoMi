"""NAOMi calcium dynamics and shared-spike fluorescence traces.

Spike input is K by T at 100 Hz, in calcium-concentration units. These routines
preserve the upstream single/double/Ca_DE equations and output semantics.
"""

import numpy as np
import scipy.signal
import scipy.optimize


def check_cal_params(cal_params=None, prot_type="GCaMP6f"):
    cal_params = {} if cal_params is None else dict(cal_params)

    defaults = {
        "ca_bind": 110,
        "ca_rest": 50e-9,
        "ind_con": 200e-6,
        "ca_dis": 290e-9,
        "ca_sat": 1,
        "sat_type": "double",
        "dt": 1 / 30.0,
        "a_bind": 3.5,
        "a_ubind": 7,
    }
    for k, v in defaults.items():
        if k not in cal_params or cal_params[k] is None:
            cal_params[k] = v

    pt = prot_type.lower()
    if "ca_amp" not in cal_params or cal_params["ca_amp"] is None:
        if pt in ["gcamp6", "gcamp6f"]:
            cal_params["ca_amp"] = 76.1251
        elif pt == "gcamp6s":
            cal_params["ca_amp"] = 54.6943
        elif pt == "gcamp7":
            cal_params["ca_amp"] = 230.917
        elif pt == "gcamp3":
            cal_params["ca_amp"] = 0.05
        else:
            cal_params["ca_amp"] = 76.1251

    if "t_on" not in cal_params or cal_params["t_on"] is None:
        if pt in ["gcamp6", "gcamp6f"]:
            cal_params["t_on"] = 0.8535
        elif pt == "gcamp6s":
            cal_params["t_on"] = 0.4526
        elif pt == "gcamp7":
            cal_params["t_on"] = 0.020137
        elif pt == "gcamp3":
            cal_params["t_on"] = 1.0
        else:
            cal_params["t_on"] = 0.8535

    if "t_off" not in cal_params or cal_params["t_off"] is None:
        if pt in ["gcamp6", "gcamp6f"]:
            cal_params["t_off"] = 98.6173
        elif pt == "gcamp6s":
            cal_params["t_off"] = 68.5461
        elif pt == "gcamp7":
            cal_params["t_off"] = 3.1295
        elif pt == "gcamp3":
            cal_params["t_off"] = 1.0
        else:
            cal_params["t_off"] = 98.6173

    if "ext_rate" not in cal_params or cal_params["ext_rate"] is None:
        if pt in ["gcamp6", "gcamp6f"]:
            cal_params["ext_rate"] = 292.3
        elif pt == "gcamp6s":
            cal_params["ext_rate"] = 299.0833
        elif pt in ["gcamp7", "gcamp3"]:
            cal_params["ext_rate"] = 265.73
        else:
            cal_params["ext_rate"] = 265.73

    return cal_params


def mk_doub_exp_ker(t_on, t_off, A, dt, dext_type="mult"):
    if dext_type == "mult":
        loc_max = np.log((t_off + t_on) / t_off) / t_on

        def dexp(z):
            return A * (1 - np.exp(-t_on * z)) * np.exp(-t_off * z)

    elif dext_type == "plus":
        A_val, B_val = (
            (A[0], A[1]) if isinstance(A, (list, tuple, np.ndarray)) and len(A) > 1 else (A, A)
        )
        t1 = min(t_off, t_on)
        t2 = max(t_off, t_on)
        loc_max = np.log(A_val * t1 / (B_val * t2)) / (t1 - t2)

        def dexp(z):
            return A_val * np.exp(-t1 * z) - B_val * np.exp(-t2 * z)

    elif dext_type == "min":
        A_val, B_val = (
            (A[0], A[1]) if isinstance(A, (list, tuple, np.ndarray)) and len(A) > 1 else (A, A)
        )

        def dexp(z):
            return np.minimum(A_val * (1 - np.exp(-t_on * z)), B_val * np.exp(-t_off * z))

        res = scipy.optimize.minimize(lambda z: -dexp(z), 0.5)
        loc_max = res.x[0]
    else:
        loc_max = np.log((t_off + t_on) / t_off) / t_on

        def dexp(z):
            return A * (1 - np.exp(-t_on * z)) * np.exp(-t_off * z)

    max_val = dexp(loc_max)
    A_sc = A[0] if isinstance(A, (list, tuple, np.ndarray)) else A
    t_max = -np.log(max_val * 1e-3 / A_sc) / t_off
    t_max = np.max(t_max)

    tz = np.arange(0, t_max + dt, dt)
    h = dexp(tz)
    return h


def sat_nonlin(CB, prot_type):
    pt = prot_type.lower()
    if pt in ["gcamp6", "gcamp6f"]:
        F0, F = 1.0, 25.2 * (1.0 / (1.0 + (290e-9 / CB) ** 2.7))
    elif pt == "gcamp6s":
        F0, F = 1.0, 27.2 * (1.0 / (1.0 + (147e-9 / CB) ** 2.45))
    elif pt == "gcamp3":
        F0, F = 2.0, 12.0 * (1.0 / (1.0 + (287e-9 / CB) ** 2.52))
    elif pt in ["ogb1", "ogb-1"]:
        F0, F = 1.0, 14.0 * (1.0 / (1.0 + 250e-9 / CB))
    elif pt in ["gcamp6-rs09", "gcamp6rs09"]:
        F0, F = 1.4, 25.0 * (1.0 / (1.0 + (520e-9 / CB) ** 3.2))
    elif pt in ["gcamp6-rs06", "gcamp6rs06"]:
        F0, F = 1.2, 15.0 * (1.0 / (1.0 + (320e-9 / CB) ** 3))
    elif pt == "jgcamp7f":
        F0, F = 1.0, 30.2 * (1.0 / (1.0 + (174e-9 / CB) ** 2.3))
    elif pt == "jgcamp7s":
        F0, F = 1.0, 40.4 * (1.0 / (1.0 + (68e-9 / CB) ** 2.49))
    elif pt == "jgcamp7b":
        F0, F = 1.0, 22.1 * (1.0 / (1.0 + (82e-9 / CB) ** 3.06))
    elif pt == "jgcamp7c":
        F0, F = 1.0, 145.6 * (1.0 / (1.0 + (298e-9 / CB) ** 2.44))
    else:
        F0, F = 1.0, 25.2 * (1.0 / (1.0 + (290e-9 / CB) ** 2.7))

    return F0 + F0 * F


def calcium_dynamics(S, cal_params=None, prot_type="GCaMP6f", over_samp=1, ext_mult=1):
    cal_params = check_cal_params(cal_params, prot_type)

    (
        ext_rate,
        ca_bind,
        ca_rest,
        ind_con,
        ca_dis,
        ca_sat,
        sat_type,
        dt,
        a_bind,
        a_ubind,
        ca_amp,
        t_on,
        t_off,
    ) = (
        ext_mult * cal_params["ext_rate"],
        cal_params["ca_bind"],
        cal_params["ca_rest"],
        cal_params["ind_con"],
        cal_params["ca_dis"],
        cal_params["ca_sat"],
        cal_params["sat_type"],
        cal_params["dt"],
        cal_params["a_bind"],
        cal_params["a_ubind"],
        cal_params["ca_amp"],
        cal_params["t_on"],
        cal_params["t_off"],
    )

    S = np.asarray(S)
    if S.ndim != 2 or S.shape[1] == 0 or not np.isfinite(S).all():
        raise ValueError("Spikes must be a finite K by T array")
    if int(over_samp) != over_samp or over_samp < 1:
        raise ValueError("over_samp must be a positive integer")
    over_samp = int(over_samp)
    if over_samp > 1:
        S_new = np.zeros((S.shape[0], S.shape[1] * over_samp))
        S_new[:, ::over_samp] = S
        S = S_new

    C = np.zeros_like(S, dtype=np.float32)
    C[:, 0] = np.maximum(ca_rest, S[:, 0])

    if sat_type == "single":
        oversampFlag = True
        a = a_bind[0] if isinstance(a_bind, (list, tuple, np.ndarray)) else a_bind
        b = a_ubind[0] if isinstance(a_ubind, (list, tuple, np.ndarray)) else a_ubind
        CB = np.zeros_like(C)

        for kk in range(1, S.shape[1]):
            C[:, kk] = (
                C[:, kk - 1]
                + dt * b * CB[:, kk - 1]
                + (-dt * ext_rate * (C[:, kk - 1] - CB[:, kk - 1] - ca_rest) + S[:, kk])
                / (1.0 + ca_bind + (ind_con * ca_dis) / (C[:, kk - 1] + ca_dis) ** 2)
            )
            if 0 <= ca_sat < 1:
                C[:, kk] = np.minimum(C[:, kk], ca_dis * ca_sat / (1.0 - ca_sat))
            CB[:, kk] = CB[:, kk - 1] + dt * (
                -b * CB[:, kk - 1] + a * (C[:, kk - 1] - CB[:, kk - 1]) * (ind_con - CB[:, kk - 1])
            )

    elif sat_type == "Ca_DE":
        oversampFlag = False
        a = (
            a_bind[0] * 100.0 * dt
            if isinstance(a_bind, (list, tuple, np.ndarray))
            else a_bind * 100.0 * dt
        )
        b = (
            a_ubind[0] * 100.0 * dt
            if isinstance(a_ubind, (list, tuple, np.ndarray))
            else a_ubind * 100.0 * dt
        )

        for kk in range(1, S.shape[1]):
            C[:, kk] = C[:, kk - 1] + (-dt * ext_rate * (C[:, kk - 1] - ca_rest) + S[:, kk]) / (
                1.0 + ca_bind + (ind_con * ca_dis) / (C[:, kk - 1] + ca_dis) ** 2
            )
            if 0 <= ca_sat < 1:
                C[:, kk] = np.minimum(C[:, kk], ca_dis * ca_sat / (1.0 - ca_sat))

        h_ca = mk_doub_exp_ker(t_on, t_off, ca_amp, dt).astype(np.float32)
        CB = np.zeros((C.shape[0], int(np.ceil(S.shape[1] / over_samp))), dtype=np.float32)
        for kk in range(C.shape[0]):
            CB[kk, :] = (np.convolve(C[kk, :] - ca_rest, h_ca, "full") + ca_rest)[
                : S.shape[1] : over_samp
            ]
        C = C[:, ::over_samp]

    elif sat_type == "double":
        oversampFlag = True
        a = np.array([a_bind, a_bind]) if isinstance(a_bind, (int, float)) else np.array(a_bind[:2])
        b = (
            np.array([a_ubind, a_ubind])
            if isinstance(a_ubind, (int, float))
            else np.array(a_ubind[:2])
        )

        CB1, CB2 = np.zeros_like(C), np.zeros_like(C)
        for kk in range(1, S.shape[1]):
            C[:, kk] = (
                C[:, kk - 1]
                + dt * (b[0] * CB1[:, kk - 1] + b[1] * CB2[:, kk - 1])
                + (
                    -dt * ext_rate * (C[:, kk - 1] - CB1[:, kk - 1] - CB2[:, kk - 1] - ca_rest)
                    + S[:, kk]
                )
                / (1.0 + ca_bind + (ind_con * ca_dis) / (C[:, kk - 1] + ca_dis) ** 2)
            )
            if 0 <= ca_sat < 1:
                C[:, kk] = np.minimum(C[:, kk], ca_dis * ca_sat / (1.0 - ca_sat))

            free_ca, free_ind = (
                C[:, kk - 1] - CB1[:, kk - 1] - CB2[:, kk - 1],
                ind_con - CB1[:, kk - 1] - CB2[:, kk - 1],
            )
            CB1[:, kk] = CB1[:, kk - 1] + dt * (-b[0] * CB1[:, kk - 1] + a[0] * free_ca * free_ind)
            CB2[:, kk] = CB2[:, kk - 1] + dt * (-b[1] * CB2[:, kk - 1] + a[1] * free_ca * free_ind)
        CB = CB1 + CB2
    else:
        raise ValueError("Unknown model!")

    if oversampFlag:
        C, CB = C[:, ::over_samp], CB[:, ::over_samp]
    bound_output = CB
    if sat_type == "single":
        a_val = a_bind[0] if isinstance(a_bind, (list, tuple, np.ndarray)) else a_bind
        b_val = a_ubind[0] if isinstance(a_ubind, (list, tuple, np.ndarray)) else a_ubind
        CB = CB + ca_rest + (b_val / a_val) * CB / (ind_con - CB)

    with np.errstate(divide="ignore", invalid="ignore"):
        fluorescence = sat_nonlin(CB, prot_type)
    return bound_output, C, fluorescence


def generate_time_traces(
    spikes,
    *,
    dt=1 / 30,
    protein="GCaMP6f",
    model="Ca_DE",
    modulation=None,
    dendrites=True,
    axons=True,
    rng=None
):
    """Generate soma/dendrite/axon activity from shared 100 Hz spike inputs.

    Unlike a spike count input, each event is supplied in concentration units
    (the upstream conventional event amplitude is 7.6e-6). No random spike
    generator is silently substituted. Returns a dict with soma/dend/bg arrays.
    """
    events = np.asarray(spikes)
    if (
        events.ndim != 2
        or events.shape[1] < 1
        or not np.isfinite(events).all()
        or np.any(events < 0)
    ):
        raise ValueError("Expected finite nonnegative K by T spike amplitudes")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive")
    if model not in ("Ca_DE", "single", "double"):
        raise ValueError("Supported models: Ca_DE, single, double")
    desired = int(np.floor(events.shape[1] / (100 * dt)))
    if desired < 1:
        raise ValueError("Input is shorter than one requested frame")
    buffered = np.concatenate([np.zeros((len(events), 500)), events], axis=1)
    if model != "Ca_DE":
        positive = buffered[buffered > 0]
        if not positive.size:
            raise ValueError("Upstream single/double trace normalization needs a positive event")
        buffered = 7.6e-6 * buffered / positive.min()
    output = {}
    for name, enabled, mult in [("soma", True, 1.0), ("dend", dendrites, 0.5), ("bg", axons, 0.25)]:
        if not enabled:
            output[name] = None
            continue
        params = {"sat_type": model, "dt": 0.01}
        if model == "single":
            params["ext_rate"] = 800 if name == "soma" else 400
        if model == "double":
            params["ext_rate"] = 800 if name == "soma" else 1000
        _, _, trace = calcium_dynamics(
            buffered, params, protein, 1, mult if model == "Ca_DE" else 1.0
        )
        trace = trace[:, 500:].astype(np.float32)
        if dt != 0.01:
            fps = int(np.floor(1 / dt))
            if fps % 10 == 0:
                down, up = 10, fps // 10
            elif abs(100 * dt - np.floor(100 * dt)) < abs(1 / dt - np.floor(1 / dt)):
                down, up = 100, fps
            else:
                down, up = int(np.floor(101 * dt)), 1
            if min(up, down, fps) < 1:
                raise ValueError("Unsupported output sampling interval")
            minimum = trace.min(axis=1, keepdims=True)
            padded = np.pad(trace.astype(np.float64), ((0, 0), (100, 100)), mode="edge")
            trace = scipy.signal.resample_poly(padded, up, down, axis=1)[:, fps:-fps]
            trace = np.maximum(trace, minimum)
        output[name] = trace[:, :desired]
    if modulation is None:
        rng = np.random.default_rng() if rng is None else rng
        modulation = rng.gamma(2.53, 0.4, size=len(events)) * (rng.random(len(events)) > 0.2)
    modulation = np.asarray(modulation, float).reshape(-1, 1)
    if modulation.shape[0] != len(events) or not np.isfinite(modulation).all():
        raise ValueError("One finite modulation per cell is required")
    for name, trace in output.items():
        if trace is not None:
            output[name] = trace * modulation
    if output["bg"] is None:
        output["bg"] = output["dend"]
    return output


def hawkes_from_network(network, frames, *, dt=0.01, uniforms=None, rng=None):
    """Conditional discrete Hawkes dynamics from an explicit upstream network.

    Network generation/continuous marked-point-process variants are separate.
    Shared uniforms validate event decisions as well as state trajectories.
    """
    matrix = np.asarray(network["A"], float)
    mu = np.asarray(network["MU"], float).ravel()
    b = np.asarray(network["B"], float).ravel()
    decay = np.asarray(network["extSc"], float).ravel()
    inhibit = np.asarray(network.get("inbSc", decay / 2), float).ravel()
    y = np.asarray(network.get("yt", np.full(len(mu), 5.0)), float).ravel().copy()
    z = np.asarray(network.get("zt", np.zeros(len(mu))), float).ravel().copy()
    if matrix.shape != (len(mu), len(mu)) or min(frames, dt) <= 0:
        raise ValueError("Invalid network or time axis")
    if uniforms is None:
        rng = np.random.default_rng() if rng is None else rng
        uniforms = rng.random((len(mu), frames))
    uniforms = np.asarray(uniforms, float)
    if uniforms.shape != (len(mu), frames) or np.any((uniforms < 0) | (uniforms >= 1)):
        raise ValueError("Invalid uniform innovations")
    single_state = np.asarray(network["B"]).dtype == np.float32
    if single_state:
        y = y.astype(np.float32)
        b = b.astype(np.float32)
        mu = mu.astype(np.float32)
    events = np.empty((len(mu), frames), bool)
    excitation = np.empty_like(uniforms)
    inhibition = np.empty_like(uniforms)
    for t in range(frames):
        if single_state:
            # MATLAB double zt minus single yt is single; rectification and
            # probability therefore use single arithmetic, including 1-exp.
            with np.errstate(over="ignore"):
                rect = np.log(
                    np.float32(1)
                    + np.exp(np.float32(3) * (z.astype(np.float32) - y + np.float32(1)))
                )
                probability = np.float32(1) - np.exp(-rect * mu * np.float32(dt))
        else:
            probability = -np.expm1(-np.logaddexp(0, 3 * (z - y + 1)) * mu * dt)
        event = uniforms[:, t] < probability
        z = np.exp(-decay * dt) * z + matrix @ event
        y = (
            (np.exp(-inhibit * dt).astype(np.float32) * y + b * event).astype(np.float32)
            if single_state
            else np.exp(-inhibit * dt) * y + b * event
        )
        events[:, t] = event
        excitation[:, t] = z
        inhibition[:, t] = y
    return {"events": events, "excitation": excitation, "inhibition": inhibition}
