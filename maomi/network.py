# Derived numerical behavior: local NAOMi checkout, Song and Charles.
"""NAOMi small-world connectivity and discrete Hawkes initialization."""

import numpy as np
from scipy.linalg import toeplitz
from ._tissue_random import RandomStream


def small_world(
    nodes,
    connections=10,
    beta=0.3,
    random_weight=0.9,
    self_excitation=4,
    locations=None,
    *,
    random_stream=None
):
    r = RandomStream() if random_stream is None else random_stream
    sizes = np.asarray(nodes).reshape(-1)
    n = int(sizes[0])
    nb = int(sizes[1]) if len(sizes) > 1 else 0
    k = int(connections)
    if locations is not None and np.asarray(locations).shape[0] == n:
        loc = np.asarray(locations)
        distance = np.sqrt(((loc[:, None] - loc[None, :]) ** 2).sum(axis=2))
        adj = np.zeros((n, n))
        for i in range(n):
            adj[i, np.argsort(distance[i], kind="stable")[:k]] = 1
    else:
        adj = toeplitz(np.r_[np.ones(k // 2), np.zeros(n - k // 2)])
    for i in range(n):
        switch = np.asarray(r.draw("rand", (k,))) < beta
        count = int(switch.sum())
        absent = np.flatnonzero(adj[i] == 0)
        present = np.flatnonzero(adj[i] == 1)
        rank = np.asarray(r.draw("randsample", (count,), high=len(absent), replace=False), int) - 1
        adj[i, absent[rank]] = 1
        # MATLAB's try/catch leaves the original edges if a true logical index
        # exceeds row degree (possible in the original Toeplitz boundary rows).
        ids = np.flatnonzero(switch)
        if not len(ids) or ids.max() < len(present):
            adj[i, present[ids]] = 0
    if nb:
        adj = np.block([[adj, np.zeros((n, nb))], [np.ones((nb, n)), np.eye(nb)]])
    adj += self_excitation * np.eye(n + nb)
    return adj * (1 - random_weight + random_weight * (0.1 + 0.9 * r.draw("rand", adj.shape)))


def initialize_hawkes(options, locations=None, *, random_stream=None, seed=None):
    """Initial network before its first event; suitable for hawkes_from_network."""
    r = RandomStream(seed=seed) if random_stream is None else random_stream
    n = int(options["K"])
    a = small_world(n, 10, 0.3, 0.9, options.get("burst_mean", 4), locations, random_stream=r)
    a = 4 * a / a.sum(axis=0).mean()
    mu = np.asarray(r.draw("gamrnd", (n,), shape_parameter=1, scale=options["rate"]), np.float32)
    b = np.asarray(r.draw("gamrnd", (n,), shape_parameter=3, scale=2), np.float32)
    np.fill_diagonal(a, np.float32(options.get("selfact", 1.26)) * b)
    decay = np.maximum(0.3, 1 + 0.3 * r.draw("randn", (n,)))
    return dict(
        A=a,
        MU=mu,
        B=b,
        extSc=decay,
        inbSc=decay / 2,
        yt=np.full(n, 5, np.float32),
        zt=np.zeros(n, np.float64),
    )
