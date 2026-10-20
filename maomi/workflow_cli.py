"""CLI for source MAT/scene import, complete acquisition and exact resume."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
from .workflow import RunConfig,run_scene
from .mat_io import load_naomi_mat,inspect_naomi_mat
from .archive import load_scene


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command",required=True)
    inspect = sub.add_parser("inspect",help="Inspect MAT variables without loading the tissue")
    inspect.add_argument("input",type=Path)
    resume = sub.add_parser("resume",help="Resume from archived scene, optics and random seeds")
    resume.add_argument("output",type=Path)
    resume.add_argument("--max-frames",type=int)
    run = sub.add_parser("run",help="Import or generate a scene and write paired data")
    run.add_argument("--input",type=Path,help="Original NAOMi MAT or MaoMi scene HDF5")
    run.add_argument("--config",type=Path,help="JSON with input, optical, tissue, run and acquisition dictionaries")
    run.add_argument("--output",type=Path,required=True)
    run.add_argument("--roi",nargs=6,type=int,metavar=("X0","X1","Y0","Y1","Z0","Z1"))
    run.add_argument("--device",default=None)
    run.add_argument("--rana",action="store_true",default=None)
    run.add_argument("--psf-field",type=Path,help="Complete XYZ FullFieldPSF or DepthPSFBank NPZ")
    run.add_argument("--count-per-axis",type=int,default=None)
    run.add_argument("--frames",type=int,help="Explicitly generate synthetic activity")
    run.add_argument("--activity-mat",type=Path)
    run.add_argument("--activity-key")
    run.add_argument("--seed",type=int,default=None)
    run.add_argument("--max-frames",type=int)
    args = p.parse_args(argv)
    if args.command=="inspect":
        print(json.dumps(inspect_naomi_mat(args.input),indent=2))
        return
    if args.command=="resume":
        result = run_scene(output=args.output,resume=True,max_frames=args.max_frames)
    else:
        cfg = json.loads(args.config.read_text(encoding="utf8")) if args.config else {}
        source = args.input or cfg.get("input")
        if source and not Path(source).is_absolute() and args.config and args.input is None:
            source = args.config.parent/source
        settings = RunConfig(**cfg.get("run",{}))
        updates = {k:getattr(args,k) for k in ("device","rana","count_per_axis","seed") if getattr(args,k) is not None}
        settings = replace(settings,**updates)
        if source:
            if str(source).lower().endswith(".mat"):
                roi = tuple(zip(args.roi[::2],args.roi[1::2])) if args.roi else cfg.get("roi")
                scene = load_naomi_mat(source,roi=roi,activity_path=args.activity_mat,activity_key=args.activity_key,
                                       optical_overrides=cfg.get("optical_overrides"))
            else:
                scene = load_scene(source)
        else:
            from .tissue_config import generate_scene
            from .optics import OpticalConfig
            scene = generate_scene(OpticalConfig(**cfg["optical"]) if "optical" in cfg else None,
                                   preset=cfg.get("preset","compact"),overrides=cfg.get("tissue"),seed=settings.seed)
        scene.scan.update(cfg.get("acquisition",{}))
        if args.frames is not None or "activity" in cfg:
            activity = cfg.get("activity",{}).copy()
            if args.frames is not None:
                activity["frames"] = args.frames
            activity.setdefault("seed",settings.seed)
            scene.generate_activity(**activity)
        addon = None
        if args.psf_field:
            import numpy as np
            from .enhanced import FullFieldPSF
            from .rana_depth import DepthPSFBank
            from .rana_addon import RanaAddon
            with np.load(args.psf_field,allow_pickle=False) as f:
                bank = str(f["schema"])=="maomi.depth_psf_bank.v1"
            field = (DepthPSFBank if bank else FullFieldPSF).load(args.psf_field)
            addon = RanaAddon(field=field,z_origin_um=scene.config.depth_um,status_window=False)
        result = run_scene(scene,settings=settings,rana=addon,output=args.output,max_frames=args.max_frames)
    print(json.dumps(dict(output=str(args.output),frames=result.frames,complete=result.complete),indent=2))
