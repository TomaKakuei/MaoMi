"""Three real RayOptics field traces -> a common-grid PSFField NPZ.

A near-axis singlet example, sampled along object-space Y. The singleton X
axis is constant; this is not a full 2-D field characterization of an objective.
"""
from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from maomi.integrations.rayoptics import demo_lens, rayoptics_psf, rayoptics_psf_field


def build_field():
    from rayoptics.raytr.opticalspec import FieldSpec
    model = demo_lens()
    spec = model["optical_spec"]
    # Absolute object heights in this millimetre lens prescription.
    spec["fov"] = FieldSpec(spec, key=["object", "height"],
                           flds=[0., .05, .10], value=.10, is_relative=False)
    model.update_model()
    pitch = rayoptics_psf(model, fft_size=128).pixel_size_um
    # Field traces can have slightly different physical sampling. Resample openly
    # onto a common grid; preserve each trace's provenance and mass estimate.
    return rayoptics_psf_field(model, [0.], [0., 50., 100.], [[0, 1, 2]],
                               pixel_size_um=pitch, shape=(128, 128), fft_size=128)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/rayoptics_field.npz"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    field = build_field()
    field.save(args.output)
    print(f"Saved {field.intensity.shape[:2]} field grid; pixel pitch {field.pixel_size_um} um")
