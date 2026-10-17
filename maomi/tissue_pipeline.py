"""Continuous NAOMi tissue simulation using completed parameter dictionaries.

Public sparse indices are one-based column-major. Optional random tapes are a
validation interface; normal use supplies only a seed and uses NumPy draws.
"""

from copy import deepcopy
import time
import numpy as np
from .tissue import params, sample_neurons, voxelize_neurons, set_fluorescence
from ._tissue_random import RandomStream
from ._tissue_vessels import simulate_vessels
from ._tissue_dendrites import grow_neuron_dendrites, grow_apical_dendrites
from ._tissue_growth import generate_background, generate_axons, sort_axons

STAGES = (
    "simulatebloodvessels",
    "sampleDenseNeurons",
    "generateNeuralVolume",
    "growNeuronDendrites",
    "growApicalDendrites",
    "setCellFluoresence",
    "generate_bgdendrites",
    "generate_axons",
    "sort_axons",
)


def simulate_tissue(vol, neur, vasc, dend, bg, axon, *, seed=None, random_tapes=None, debug=False):
    """Return (neurons, metadata), including per-stage wall times.

    Parameters must contain NAOMi check_* defaults (see the runnable example
    configuration). This API does not silently infer an optical padding size.
    vasc_sz must be explicitly supplied in physical microns.
    """
    v, n, va, d, b, a = [params(deepcopy(x)) for x in (vol, neur, vasc, dend, bg, axon)]
    shared = RandomStream(seed=seed)
    times = {}
    counts = {}

    def call(name, fn, *args):
        r = shared if random_tapes is None else RandomStream(random_tapes[name])
        t = time.perf_counter()
        out = fn(*args, random_stream=r)
        times[name] = time.perf_counter() - t
        r.finish()
        counts[name] = r.cursor if random_tapes is not None else None
        return out

    if va.flag:
        vessels, va, full = call(STAGES[0], simulate_vessels, v, va)
    else:
        vessels = np.zeros(
            tuple(((np.asarray(v.vol_sz) + [0, 0, v.vol_depth]) * v.vres).astype(int)), bool
        )
        full = np.zeros(tuple((np.asarray(v.vasc_sz) * v.vres).astype(int)), bool)
    locs, cells, nuclei, tri, angles = call(STAGES[1], sample_neurons, n, v, vessels)
    v.N_neur = cells.shape[2]
    t = time.perf_counter()
    soma, volume, gp_nuc, gp_soma = voxelize_neurons(n, v, locs, cells, nuclei, vessels)
    times[STAGES[2]] = time.perf_counter() - t
    if random_tapes is not None:
        RandomStream(random_tapes[STAGES[2]]).finish()
        counts[STAGES[2]] = 0
    vessels = vessels.copy()
    depth = int(v.vol_depth * v.vres)
    vessels[:, :, depth : depth + soma.shape[2]][soma > 0] = False
    nuc0 = [[np.asarray(ids, dtype=np.int64) - 1, value] for ids, value in gp_nuc]
    soma0 = [[np.asarray(ids, dtype=np.int64) - 1] for ids in gp_soma]
    labels, apical, d, soma0 = call(
        STAGES[3], grow_neuron_dendrites, v, d, soma, vessels, locs, nuc0, soma0, angles
    )
    labels, apical, d = call(STAGES[4], grow_apical_dendrites, v, d, labels, apical, nuc0, soma0)
    gp_soma = [[np.asarray(ids, dtype=np.int32) + 1 for ids in row] for row in soma0]
    gp_vals, volume = call(STAGES[5], set_fluorescence, v, n, d, labels, soma, apical, locs, volume)
    if b.flag:
        labels, volume, v, gp_vals, locs = call(
            STAGES[6], generate_background, v, b, d, volume, labels, gp_vals, gp_nuc, locs
        )
    background = []
    bg_proc = []
    if a.flag:
        volume, background, a, _ = call(
            STAGES[7], generate_axons, v, a, volume, labels, gp_vals, gp_nuc
        )
        a.N_proc = len(gp_vals)
        bg_proc = call(STAGES[8], sort_axons, v, a, background, locs * v.vres)
    out = dict(
        neur_vol=volume,
        gp_nuc=gp_nuc,
        gp_soma=gp_soma,
        gp_vals=gp_vals,
        neur_ves=vessels,
        bg_proc=bg_proc,
        neur_ves_all=full,
        locs=locs,
    )
    if debug:
        out.update(
            neur_num_AD=apical,
            neur_soma=soma,
            neur_num=labels,
            gp_bgvals=background,
            Vcell=cells,
            Vnuc=nuclei,
            Tri=tri,
        )
    meta = dict(
        timing_seconds=times,
        random_calls_replayed=counts,
        parameters=dict(
            vol=vars(v), neur=vars(n), vasc=vars(params(va)), dend=vars(d), bg=vars(b), axon=vars(a)
        ),
    )
    return out, meta
