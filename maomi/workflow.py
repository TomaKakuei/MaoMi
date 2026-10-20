"""One scene workflow for CPU/CUDA, standard/RANA, and resumable paired movies."""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import time
import numpy as np
from .scene import Scene


def select_device(device="auto"):
    if device == "auto":
        import importlib.util
        import torch
        return "cuda" if torch.cuda.is_available() and importlib.util.find_spec("cupy") is not None else "cpu"
    if device == "cpu" or str(device).startswith("cuda"):
        return str(device)
    raise ValueError("device must be auto, cpu or cuda[:index]")


def render_static(volume,optics,*,device="auto"):
    """Render from reusable optics, including RANA and incoherent beam sums."""
    from .scanning import zstack
    device = select_device(device)
    if "channels" in optics:
        return sum(optics["beam_weights"][k]*render_static(volume,v,device=device)
                   for k,v in optics["channels"].items())
    return zstack(volume,optics["psf"],mask=optics.get("mask"),
                  collection_mask=optics.get("colmask"),rana=optics.get("rana"),
                  device=device).cpu().numpy()


def prepare_optics(config, *, device="auto", rana=False, vessels=None, raw_fields=None,
                   scatter=True, custom_phase=None, phase_units="radians", phase_aperture="objective",
                   input_pupil=None, beam_weights=None, tails=True, count_per_axis=65, batch_size=32):
    """Compute optics once; no fluorescence volume or static image is needed."""
    from .fixed import run_fixed, fixed_locations
    from .optics import prepare_fields, pupil, propagate, collection_mask
    from .rana_addon import attach_rana
    device = select_device(device)
    if device.startswith("cuda"):
        return run_fixed(config,None,device=device,rana=rana,render_image=False,vessels=vessels,
            raw_fields=raw_fields,scatter=scatter,custom_phase=custom_phase,phase_units=phase_units,
            phase_aperture=phase_aperture,input_pupil=input_pupil,beam_weights=beam_weights,
            tails=tails,count_per_axis=count_per_axis,batch_size=batch_size)
    if input_pupil is not None and custom_phase is not None:
        raise ValueError("Supply input_pupil or custom_phase")
    started = time.perf_counter()
    raw = prepare_fields(config,vessels,scatter=scatter,device="cpu") if raw_fields is None else raw_fields
    points = fixed_locations(config,count_per_axis)
    if isinstance(input_pupil,dict):
        weights = {k:1. for k in input_pupil} if beam_weights is None else dict(beam_weights)
        if not weights or set(weights)!=set(input_pupil) or any(not np.isfinite(v) or v<0 for v in weights.values()) or sum(weights.values())<=0:
            raise ValueError("Invalid incoherent beam weights")
        channels = {k:prepare_optics(config,device=device,rana=False,vessels=vessels,raw_fields=raw,
            input_pupil=v,tails=tails,count_per_axis=count_per_axis) for k,v in input_pupil.items()}
        result = dict(channels=channels,beam_weights=weights,points_um=points,
                      unique_optical_paths=len(points)*len(channels),unique_optical_positions=len(points))
    else:
        if beam_weights is not None:
            raise ValueError("beam_weights requires a mapping of pupils")
        u = pupil(config,custom_phase=custom_phase,phase_units=phase_units,phase_aperture=phase_aperture,device=device) if input_pupil is None else input_pupil
        result = propagate(config,raw,points,uin=u,device=device)
        result["colmask"] = collection_mask(vessels,config) if vessels is not None else np.ones_like(result["mask"])
        if not tails:
            result.pop("psfT",None)
            result.pop("psfB",None)
    result["settings"] = dict(device=device,seed=config.seed,scatter=scatter,count_per_axis=count_per_axis,
                              mask_mode="matlab_v4",cpu_tail_calculation="computed; returned only when requested")
    result["total_seconds"] = time.perf_counter()-started
    result["timing_seconds"] = dict(optics=result["total_seconds"],static_scan=0.)
    if rana is not False and rana is not None:
        result = attach_rana(result,rana,config=config,device=device)
    return result


@dataclass(frozen=True)
class RunConfig:
    device: str = "auto"
    rana: bool = False
    count_per_axis: int = 65
    batch_size: int = 32
    scatter: bool = True
    tails: bool = True
    add_noise: bool = True
    seed: int = 0
    chunk_frames: int = 64

    def __post_init__(self):
        for name in ("count_per_axis","batch_size","chunk_frames"):
            value = getattr(self,name)
            if not isinstance(value,(int,np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.seed,(int,np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def run_scene(scene=None, *, settings=None, optics=None, rana=None, motion=None,
              output=None, resume=False, max_frames=None, **optical_options):
    """Run or resume a full acquisition. Resume uses archived inputs exactly.

    With output=None collect XYT arrays. With output=folder write atomic chunks
    and all GT inputs; max_frames is an optional bounded checkpoint interval.
    """
    from .archive import save_scene,load_scene,save_optics,load_optics,jsonable
    from .movie import scan_volume,iter_scan_volume,sample_motion
    from .dataset import MovieDataset,write_chunks,_sha256
    if resume and output is None:
        raise ValueError("Resume requires an output directory")
    path = Path(output) if output is not None else None
    if resume:
        if scene is not None or optics is not None or optical_options or rana is not None or motion is not None:
            raise ValueError("Resume with scene=None; source inputs and optics are restored from the archive")
        previous = json.loads((path/"run.json").read_text(encoding="utf8"))
        from . import __version__
        if previous.get("schema")!="maomi.run.v1" or previous.get("maomi_version",__version__)!=__version__:
            raise ValueError("Resume with the MaoMi version that created this run")
        if settings is not None and asdict(settings) != previous["settings"]:
            raise ValueError("RunConfig differs from the saved run")
        settings = RunConfig(**previous["settings"])
        device = previous.get("selected_device",select_device(settings.device))
        for name,digest in previous.get("archive_sha256",{}).items():
            if name not in ("scene.h5","optics.h5") or _sha256(path/name)!=digest:
                raise ValueError("Archived scene/optics checksum mismatch")
        scene = load_scene(path/"scene.h5")
        optics = load_optics(path/"optics.h5",device=device)
        motion = previous["motion"]
        start = MovieDataset(path,allow_incomplete=True,verify=True).frames
    else:
        if not isinstance(scene,Scene):
            raise TypeError("Supply a Scene or use resume=True")
        scene.validate()
        if scene.activity is None:
            raise ValueError("No activity supplied; import traces or call scene.generate_activity(...) explicitly")
        needed = scene.metadata.get("required_optical_overrides",[])
        if optics is None and needed:
            if "input_pupil" in needed and "input_pupil" not in optical_options:
                raise ValueError("Imported non-Gaussian optics require an explicit input_pupil; original parameters are in scene.metadata")
            if "custom_phase or input_pupil" in needed and not ({"custom_phase","input_pupil"} & optical_options.keys()):
                raise ValueError("Imported spatial Zernike callbacks require an explicit custom_phase or input_pupil")
        settings = settings or RunConfig()
        device = select_device(settings.device)
        if settings.chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        if path is not None:
            if path.exists():
                raise FileExistsError("Choose a new output folder or resume the existing run")
            path.mkdir(parents=True)
        if optics is None:
            optics = prepare_optics(scene.config,device=device,
                rana=settings.rana if rana is None else rana,vessels=scene.vessels,
                count_per_axis=settings.count_per_axis,batch_size=settings.batch_size,
                scatter=settings.scatter,tails=settings.tails,**optical_options)
        elif rana is not None or settings.rana:
            from .rana_addon import attach_rana
            optics = attach_rana(optics,settings.rana if rana is None else rana,config=scene.config,device=device)
        p = dict(scan_buff=10,scan_avg=2,sfrac=2,motion=True,zoffset=0)
        p.update(scene.scan)
        channels = list(optics["channels"].values()) if "channels" in optics else [optics]
        depths = {c["rana"].kernel_shape[2] if c.get("rana") is not None else c["psf"].shape[2] for c in channels}
        if len(depths) != 1:
            raise ValueError("All channels must share the acquisition depth")
        if motion is None:
            motion = sample_motion(scene.shape,next(iter(depths)),scene.activity["soma"].shape[1],
                buffer=int(p["scan_buff"]),zoffset=p["zoffset"],enabled=p["motion"],
                rng=np.random.default_rng(settings.seed))
        start = 0
        if path is not None:
            save_scene(scene,path/"scene.h5")
            save_optics(optics,path/"optics.h5")
            from . import __version__
            previous = dict(schema="maomi.run.v1",maomi_version=__version__,settings=asdict(settings),selected_device=device,
                            config=asdict(scene.config),motion=motion,source=scene.metadata,
                            optical_settings=optics.get("settings",{}),
                            archive_sha256={name:_sha256(path/name) for name in ("scene.h5","optics.h5")},
                            targets=dict(clean="expected acquisition including motion and background",
                                         noisy="detector observation",density="latent fluorescence",
                                         cells="one-based sparse component labels"))
            (path/"run.json").write_text(json.dumps(jsonable(previous),indent=2,allow_nan=False),encoding="utf8")
    scan = dict(scene.scan,chunk_frames=settings.chunk_frames)
    args = (scene.neurons,optics,scene.activity,scan,scene.noise,scene.spikes,scene.tpm)
    kw = dict(motion=motion,add_noise=settings.add_noise,start_frame=start,frame_seed=settings.seed)
    if path is None:
        if max_frames is not None:
            raise ValueError("max_frames requires an output directory")
        return scan_volume(*args,**kw)
    expected = scene.activity["soma"].shape[1]
    if start < expected:
        write_chunks(iter_scan_volume(*args,**kw),path,frames=expected,
                     chunk_frames=settings.chunk_frames,start=start,max_frames=max_frames)
    return MovieDataset(path,allow_incomplete=True)
