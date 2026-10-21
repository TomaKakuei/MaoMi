"""A solver-independent PSF provider; exports a field for the shared CLI.

This analytic astigmatic field demonstrates the provider contract. Substitute
your calibrated effective-intensity solver in kernel() for an instrument model.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
from maomi.rana import DepthPSFBank


@dataclass(frozen=True)
class AstigmaticProvider:
    # Matches `maomi run/render --demo`: 16 x 16 x 10 um, lower depth 10 um.
    x_um: tuple = (0., 8., 16.)
    y_um: tuple = (0., 8., 16.)
    depths_um: tuple = (10., 15., 20.)
    voxel_size_um: tuple = (1., 1., 1.)

    def kernel(self, x, y, depth):
        offsets = [(np.arange(5)-2)*pitch for pitch in self.voxel_size_um]
        xx, yy, zz = np.meshgrid(*offsets, indexing='ij')
        sx, sy = .65+.35*x/16, .65+.35*y/16
        sz = 1.+.4*(depth-10)/10
        # Already effective multiphoton response; no later squaring/fitting.
        return np.exp(-.5*((xx/sx)**2+(yy/sy)**2+(zz/sz)**2))

    def build_field(self):
        return DepthPSFBank.from_provider(self.kernel, self.x_um, self.y_um,
            self.depths_um, voxel_size_um=self.voxel_size_um, wavelength_nm=920,
            operator='excitation', provenance={'model': 'analytic astigmatic example',
                                               'units': 'relative effective response'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/custom_bank.npz'))
    args = parser.parse_args()
    AstigmaticProvider().build_field().save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
