"""Random input replay for stage-wise MATLAB/Python parity checks."""

import numpy as np


class RandomStream:
    def __init__(self, tape=None, seed=None):
        self.tape = None if tape is None else list(np.atleast_1d(tape))
        self.cursor = 0
        self.rng = np.random.default_rng(seed)

    def draw(self, kind, shape=None, **kw):
        if self.tape is not None:
            if self.cursor >= len(self.tape):
                raise ValueError(f"Random tape exhausted at {self.cursor}: {kind}")
            row = self.tape[self.cursor]
            if row["kind"] != kind:
                raise ValueError(
                    f"Random call {self.cursor}: expected MATLAB {row['kind']}, requested {kind}"
                )
            self.cursor += 1
            a = np.asarray(row["value"])
            if shape is None:
                return a.item() if a.size == 1 else a.copy()
            shape = tuple(np.atleast_1d(shape).astype(int))
            if a.size != int(np.prod(shape)):
                raise ValueError(f"Random call {self.cursor-1}: shape {a.shape} vs {shape}")
            return a.reshape(shape, order="F").copy()
        if kind == "rand":
            return self.rng.random(shape)
        if kind == "randn":
            return self.rng.standard_normal(shape)
        if kind == "mvnrnd":
            return self.rng.multivariate_normal(kw["mean"], kw["cov"])
        if kind == "randsample" and kw.get("replace") is False:
            return self.rng.choice(int(kw["high"]), size=shape, replace=False) + 1
        if kind in ("randsample", "randi"):
            return self.rng.integers(1, int(kw["high"]) + 1, size=shape)
        if kind == "randperm":
            return self.rng.permutation(int(kw["high"])) + 1
        if kind == "normrnd":
            return self.rng.normal(kw["mean"], kw["sd"], size=shape)
        if kind == "gamrnd":
            return self.rng.gamma(kw["shape_parameter"], kw["scale"], size=shape)
        raise NotImplementedError(kind)

    def random(self, size=None):
        return self.draw("rand", size)

    def rand(self, *shape):
        return self.draw("rand", shape or None)

    def randn(self, *shape):
        return self.draw("randn", shape or None)

    def multivariate_normal(self, mean, cov):
        return self.draw("mvnrnd", np.shape(mean), mean=mean, cov=cov)

    def rank(self, n):
        return int(self.draw("randsample", high=n))

    def integer(self, n):
        return int(self.draw("randi", high=n))

    def normal(self, loc=0, scale=1, size=None):
        return self.draw("normrnd", size, mean=loc, sd=scale)

    def finish(self):
        if self.tape is not None and self.cursor != len(self.tape):
            raise ValueError(f"Unused random calls: {len(self.tape)-self.cursor}")
