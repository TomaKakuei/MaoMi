"""Common anatomy, activity and acquisition inputs for all MaoMi backends."""
from dataclasses import dataclass, field
import numpy as np
from .optics import OpticalConfig


@dataclass
class Scene:
    neurons: dict
    config: OpticalConfig
    activity: dict | None = None
    vessels: object = None
    scan: dict = field(default_factory=dict)
    noise: dict = field(default_factory=dict)
    spikes: dict = field(default_factory=dict)
    tpm: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    @property
    def shape(self):
        return tuple(self.neurons["neur_vol"].shape)

    def validate(self):
        expected = tuple(np.rint(np.asarray(self.config.volume_um)*self.config.vres).astype(int))
        if self.shape != expected or len(self.shape) != 3:
            raise ValueError(f"Tissue XYZ shape {self.shape} differs from optical grid {expected}")
        if "gp_vals" not in self.neurons:
            raise ValueError("Sparse gp_vals are required for functional movies")
        return self

    def generate_activity(self, frames=100, *, fps=30., rate_hz=.5, seed=0):
        """Explicit synthetic spike/AR fluorescence recipe, replacing activity."""
        from .temporal import ARTraceFilter
        if frames < 1 or fps <= 0 or rate_hz < 0:
            raise ValueError("Positive frames/fps and nonnegative rate required")
        rng = np.random.default_rng(seed)
        n, nb = len(self.neurons["gp_vals"]), len(self.neurons.get("bg_proc", []))
        counts = rng.poisson(rate_hz/fps, (n, int(frames)))
        soma = 1 + ARTraceFilter(n, dt=1/fps, tau_decay=.4, tau_rise=.06).process(counts)
        self.activity = dict(soma=soma, dend=soma.copy(), bg=np.ones((nb, int(frames))))
        self.spikes = dict(self.spikes, dt=1/fps)
        self.metadata = dict(self.metadata, activity_recipe=dict(
            model="Poisson spikes and AR fluorescence", seed=seed, rate_hz=rate_hz, fps=fps))
        self.neurons["spike_counts"] = counts
        return self
