"""RayOptics XYZ two-photon field -> RANA movie, or pupil -> scattering -> RANA."""
import argparse
from pathlib import Path
import numpy as np
from maomi import Scene,RunConfig,run_scene,RanaAddon,OpticalConfig
from maomi.integrations.rayoptics import demo_lens
from maomi.integrations.rayoptics_volume import rayoptics_volume_field,rayoptics_native_pupil
from rana_movie import inputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route",choices=["field","pupil"],default="field")
    p.add_argument("--device",default="cpu")
    p.add_argument("--output",type=Path,default=Path("artifacts/rayoptics_movie"))
    args = p.parse_args()
    cfg,neurons,activity,motion,counts = inputs()
    model = demo_lens(wavelength_nm=cfg.wavelength_um*1000)
    scene = Scene(neurons,cfg,activity,scan=dict(scan_buff=2,scan_avg=1,sfrac=2))
    scene.neurons["spike_counts"] = counts
    settings = RunConfig(device=args.device,rana=True,count_per_axis=3,chunk_frames=4,seed=31)
    if args.route=="field":
        field = rayoptics_volume_field(model,(0,24),(0,24),np.zeros((2,2),int),
            voxel_size_um=(1,1,1),shape=(8,8,4),num_rays=9,fft_size=64,
            photon_order=2,specimen_index=cfg.n)
        # A single on-axis prescription is deliberately repeated for this minimal
        # workflow. Supply actual field_indices for field-dependent aberrations.
        result = run_scene(scene,settings=settings,motion=motion,output=args.output,
                           rana=RanaAddon(field=field,status_window=False))
    else:
        pupil = rayoptics_native_pupil(model,cfg,num_rays=17)
        result = run_scene(scene,settings=settings,motion=motion,output=args.output,input_pupil=pupil)
    print(dict(route=args.route,frames=len(result),complete=result.complete))


if __name__=="__main__":
    main()
