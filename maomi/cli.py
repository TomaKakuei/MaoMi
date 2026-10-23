"""One command set for MaoMi, with RANA as an optional acquisition add-on."""
import argparse
from dataclasses import replace
import json
from pathlib import Path


def build_parser():
    from . import __version__
    parser = argparse.ArgumentParser(prog="maomi", description=__doc__,
        epilog="Use maomi COMMAND --help for options. Add --rana to run or render; resume restores it automatically.")
    parser.add_argument("--version", action="version", version=f"MaoMi {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("run", "Generate a clean/noisy movie and ground truth"),
                            ("render", "Render a noiseless static XYZ volume")):
        command = commands.add_parser(name, help=help_text, description=help_text +
            ". Standard and RANA share these inputs, optics settings and output conventions.")
        source = command.add_argument_group("scene")
        inputs = source.add_mutually_exclusive_group()
        inputs.add_argument("--input", type=Path, help="Original NAOMi MAT or archived scene.h5")
        inputs.add_argument("--demo", action="store_true", help="Small built-in XYZ phantom; 3x3 optical sites, no tissue extra needed")
        if name == "render":
            inputs.add_argument("--volume", type=Path, help="XYZ fluorescence NPY; requires optical geometry in --config")
        source.add_argument("--config", type=Path, help="JSON scene, optics, run and acquisition (optional acquisition.timing for time-resolved movies); optional rana provider")
        source.add_argument("--roi", nargs=6, type=int, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                            help="Zero-based, half-open voxel bounds for a MAT input")
        optical = command.add_argument_group("execution and optics")
        optical.add_argument("--device", help="auto (default), cpu or cuda[:index]")
        optical.add_argument("--seed", type=int, help="Override the configured random seed")
        optical.add_argument("--count-per-axis", type=int, help="Native optical sites per axis; default 65, demo 3")
        optical.add_argument("--batch-size", type=int, help="CUDA propagation batch size; default 32")
        optical.add_argument("--focal-only", action="store_true", default=None, help="Omit native top/bottom tail descriptors")
        optical.add_argument("--rana", action="store_true", default=None,
                             help="Enable RANA (Ray Aware Nonuniform Acquisition) with full local PSFs")
        optical.add_argument("--psf-field", type=Path,
                             help="RANA focal provider: XYZ FullFieldPSF or DepthPSFBank NPZ; requires --rana or run.rana=true")
        optical.add_argument("--no-status-window", action="store_true", help="Disable the optional desktop status indicator")
        command.add_argument("--output", type=Path, required=True, help="New output DIRECTORY, never a filename prefix")
        if name == "run":
            acquisition = command.add_argument_group("movie")
            acquisition.add_argument("--frames", type=int, help="Explicitly generate synthetic activity for this many frames")
            acquisition.add_argument("--activity-mat", type=Path, help="Activity stored in a separate MAT file")
            acquisition.add_argument("--activity-key", help="MAT activity variable name")
            acquisition.add_argument("--max-frames", type=int, help="Write at most this many frames, then continue with resume")
    resume = commands.add_parser("resume", help="Continue a movie using its saved standard/RANA configuration",
        description="Restore scene, local PSFs, motion, seeds and device from the movie directory. No mode flags needed.")
    resume.add_argument("output", type=Path)
    resume.add_argument("--max-frames", type=int, help="Write at most this many additional frames")
    inspect = commands.add_parser("inspect", help="Inspect MAT variables, a scene, a PSF field or output metadata")
    inspect.add_argument("input", type=Path)
    return parser


def _relative(value, config_path):
    path = Path(value)
    return config_path.parent/path if config_path and not path.is_absolute() else path


def _load_field(path):
    import numpy as np
    from .rana import DepthPSFBank, FullFieldPSF
    with np.load(path, allow_pickle=False) as data:
        schema = str(data["schema"])
    return (DepthPSFBank if schema == "maomi.depth_psf_bank.v1" else FullFieldPSF).load(path)


def _phantom(config, seed):
    """Small complete Scene, deliberately without morphology dependencies."""
    import numpy as np
    from .optics import OpticalConfig
    from .scene import Scene
    cfg = OpticalConfig(**config) if config else OpticalConfig(
        volume_um=(16., 16., 10.), depth_um=10., psf_um=(4., 4., 4.), tail_um=5., seed=seed)
    shape = tuple(np.rint(np.asarray(cfg.volume_um)*cfg.vres).astype(int))
    grid = np.meshgrid(*[np.arange(n)/n for n in shape], indexing="ij")
    volume = np.exp(-sum((q-c)**2 for q, c in zip(grid, (.4, .5, .5)))/.025)
    return _volume_scene(volume, cfg, scan=dict(scan_buff=1, scan_avg=1, sfrac=1, motion=False))


def _volume_scene(volume, config, **options):
    import numpy as np
    from .scene import Scene
    volume = np.asarray(volume)
    if volume.ndim != 3 or np.iscomplexobj(volume) or not np.isfinite(volume).all() or (volume < 0).any():
        raise ValueError("Supply a finite nonnegative XYZ fluorescence volume")
    weights = volume.ravel(order="F")
    ids = np.flatnonzero(weights > 0)
    return Scene(dict(neur_vol=volume, gp_vals=[(ids+1, weights[ids], np.ones(len(ids), bool))],
                      bg_proc=[]), config, **options).validate()


def _inputs(args):
    from .workflow import RunConfig
    from .archive import load_scene
    from .mat_io import load_naomi_mat
    cfg = json.loads(args.config.read_text(encoding="utf8")) if args.config else {}
    allowed = {"input", "optical", "tissue", "preset", "run", "acquisition", "activity", "roi",
               "optical_overrides", "rana", "noise", "spikes", "tpm"}
    unknown = set(cfg)-allowed
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    settings = RunConfig(**dict({"count_per_axis": 3} if args.demo else {}, **cfg.get("run", {})))
    changes = {key: getattr(args, key) for key in ("device", "rana", "count_per_axis", "batch_size", "seed")
               if getattr(args, key) is not None}
    if args.focal_only:
        changes["tails"] = False
    settings = replace(settings, **changes)
    source = args.input or (_relative(cfg["input"], args.config) if cfg.get("input") else None)
    if (args.demo or getattr(args, "volume", None)) and source:
        raise ValueError("Choose a configured input or --demo/--volume, not both")
    if (args.psf_field or cfg.get("rana")) and not settings.rana:
        raise ValueError("RANA provider settings require --rana (or run.rana=true in JSON)")
    if args.output.exists():
        raise FileExistsError("Choose a new --output directory or use maomi resume DIRECTORY for a movie")
    if args.roi is not None and (not source or Path(source).suffix.lower() != ".mat"):
        raise ValueError("--roi requires a MAT input")
    if (getattr(args, "activity_mat", None) or getattr(args, "activity_key", None)) and (not source or Path(source).suffix.lower() != ".mat"):
        raise ValueError("--activity-mat/--activity-key require a MAT input")
    if source:
        if Path(source).suffix.lower() == ".mat":
            roi = tuple(zip(args.roi[::2], args.roi[1::2])) if args.roi is not None else cfg.get("roi")
            scene = load_naomi_mat(source, roi=roi, activity_path=getattr(args, "activity_mat", None),
                activity_key=getattr(args, "activity_key", None), optical_overrides=cfg.get("optical_overrides"))
        else:
            scene = load_scene(source)
    elif args.demo:
        scene = _phantom(cfg.get("optical"), settings.seed)
    elif getattr(args, "volume", None):
        import numpy as np
        from .optics import OpticalConfig
        if "optical" not in cfg:
            raise ValueError("--volume requires an optical configuration with volume_um, vres and depth_um")
        scene = _volume_scene(np.load(args.volume, allow_pickle=False), OpticalConfig(**cfg["optical"]))
    elif args.config:
        from .tissue_config import generate_scene
        from .optics import OpticalConfig
        scene = generate_scene(OpticalConfig(**cfg["optical"]) if "optical" in cfg else None,
            preset=cfg.get("preset", "compact"), overrides=cfg.get("tissue"), seed=settings.seed)
    else:
        raise ValueError("Supply --input, --config or --demo")
    for key in ("acquisition", "noise", "spikes", "tpm"):
        getattr(scene, "scan" if key == "acquisition" else key).update(cfg.get(key, {}))
    if args.command == "run":
        activity = cfg.get("activity", {}).copy()
        if args.frames is not None:
            activity["frames"] = args.frames
        elif args.demo and not activity:
            activity["frames"] = 8
        if activity:
            activity.setdefault("seed", settings.seed)
            scene.generate_activity(**activity)
    addon = False
    if settings.rana:
        from .rana import RanaAddon, RanaConfig
        spec = dict(cfg.get("rana", {}))
        field_path = args.psf_field or (_relative(spec.pop("field"), args.config) if spec.get("field") else None)
        # A CLI field overrides a configured field, but never a physics recipe.
        spec.pop("field", None)
        physics = spec.pop("physics", None)
        if field_path and physics:
            raise ValueError("Choose one RANA focal provider: psf-field or rana.physics")
        field = _load_field(field_path) if field_path else RanaConfig.from_dict(physics) if physics else None
        spec.setdefault("z_origin_um", scene.config.depth_um)
        spec["status_window"] = False  # One UI lifecycle around the complete command.
        addon = RanaAddon(field=field, **spec)
    return scene, settings, addon


def _inspect(path):
    if path.is_dir():
        name = "run.json" if (path/"run.json").exists() else "render.json"
        result = json.loads((path/name).read_text(encoding="utf8"))
        if (path/"dataset.json").exists():
            result["dataset"] = json.loads((path/"dataset.json").read_text(encoding="utf8"))
        return result
    if path.suffix.lower() == ".mat":
        from .mat_io import inspect_naomi_mat
        return inspect_naomi_mat(path)
    if path.suffix.lower() == ".npz":
        from .rana import DepthPSFBank
        field = _load_field(path)
        fields = field.fields if isinstance(field, DepthPSFBank) else (field,)
        return dict(kind=type(field).__name__, voxel_size_um=field.voxel_size_um,
                    depths_um=field.depths_um if isinstance(field, DepthPSFBank) else None,
                    fields=[dict(shape=f.kernels.shape, x_um=f.x_um, y_um=f.y_um,
                                 operator=f.operator, provenance=f.provenance) for f in fields])
    from .archive import load_scene
    scene = load_scene(path)
    return dict(kind="Scene", shape=scene.shape, activity=scene.activity is not None, metadata=scene.metadata)


def main(argv=None):
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if any(value == "--mode" or value.startswith("--mode=") for value in argv):
        parser.error("--mode has been removed. Use maomi run/render ... --rana. See docs/CLI.md for migration.")
    args = parser.parse_args(argv)
    try:
        from .archive import jsonable
        from .workflow import run_scene, render_scene
        if args.command == "inspect":
            result = _inspect(args.input)
        elif args.command == "resume":
            data = run_scene(output=args.output, resume=True, max_frames=args.max_frames)
            result = dict(output=str(args.output), frames=data.frames, complete=data.complete)
        else:
            scene, settings, addon = _inputs(args)
            from .rana_ui import RanaStatus
            with RanaStatus(settings.rana and not args.no_status_window):
                if args.command == "run":
                    data = run_scene(scene, settings=settings, rana=addon, output=args.output, max_frames=args.max_frames)
                    result = dict(output=str(args.output), frames=data.frames, complete=data.complete)
                else:
                    rendered = render_scene(scene, settings=settings, rana=addon, output=args.output)
                    result = dict(output=str(args.output), **rendered["metadata"])
        print(json.dumps(jsonable(result), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
