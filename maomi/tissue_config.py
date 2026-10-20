"""Complete editable tissue presets derived from NAOMi check_* defaults."""
from copy import deepcopy
import numpy as np
from .optics import OpticalConfig


def tissue_parameters(config=None,*,preset="compact",overrides=None):
    """Return all six dictionaries for simulate_tissue.

    naomi uses upstream morphology defaults at the requested volume size.
    compact explicitly reduces process counts/branches for a short tutorial;
    its vessel, background and axon stages are off and remain individually
    configurable. Neither preset silently changes the optical volume.
    """
    cfg = config or OpticalConfig(volume_um=(32.,32.,20.),depth_um=10.,psf_um=(4.,4.,4.),tail_um=5.)
    if preset not in ("compact","naomi"):
        raise ValueError("preset must be compact or naomi")
    size = np.array(cfg.volume_um)
    if (size<=0).any() or not np.allclose(size,np.round(size)) or not np.allclose(size*cfg.vres,np.round(size*cfg.vres)):
        raise ValueError("Tissue growth needs integer micrometre extents on the requested voxel grid")
    v = dict(vol_sz=size,vasc_sz=np.array(cfg.parameters()[0].vasc_sz),vres=cfg.vres,
             vol_depth=cfg.depth_um,min_dist=16.,N_neur=max(1,int(np.ceil(1e-4*np.prod(size)))),
             N_den=max(0,int(np.ceil(.002*np.prod(size[:2])))),N_den2=0,N_bg=1_000_000,
             neur_density=1e5,AD_density=2e3,dendrite_tau=5.,verbose=0)
    n = dict(n_samps=200,l_scale=90.,p_scale=1000.,avg_rad=5.9,nuc_rad=[5.65,2.5],
             max_ang=20.,plot_opt=False,dendrite_tau=50.,nuc_fluorsc=0.,min_thic=[.4,.4],
             eccen=[.35,.35,.5],exts=[.75,1.7],nexts=[.5,1.],neur_type="pyr",fluor_dist=[1.,.2])
    d = dict(dtParams=[40,150,50,1,10],atParams=[6,5,5,5,1],atParams2=[1,5,5,5,4],
             dweight=10.,bweight=5.,thicknessScale=.5,weightScale=[150,1,.8],
             dims=[60,60,60],dimsSS=[5,5,5],rallexp=1.5)
    va = dict(flag=1,ves_shift=[5,15,5],depth_vasc=200.,depth_surf=15.,distWeightScale=2.,
              randWeightScale=.1,cappAmpScale=.5,cappAmpZscale=.5,vesSize=[15,9,2],
              vesFreq=[125,200,50],sourceFreq=1000.,vesNumScale=.2,sepweight=.75,distsc=4.,
              node_params=dict(maxit=25,lensc=50.,varsc=15.,mindist=10.,varpos=5.,dirvar=np.pi/8,branchp=.02,vesrad=25.))
    b = dict(flag=1,distsc=.5,fillweight=100.,maxlength=200.,minlength=10.,maxdist=100.,maxel=1)
    a = dict(flag=1,distsc=.5,fillweight=100.,maxlength=200.,minlength=10.,maxdist=100.,maxel=8,
             varfill=.3,maxvoxel=6,padsize=20,numbranches=20,varbranches=5,maxfill=.5,N_proc=10,l=25.,rho=.1)
    if preset=="compact":
        v.update(N_neur=1,N_den=0,N_den2=0,N_bg=0)
        d.update(dtParams=[3,8,4,1,0],atParams=[1,3,3,3,0],dims=[8,8,8],dimsSS=[2,2,2])
        va["flag"] = b["flag"] = a["flag"] = 0
    out = dict(vol=v,neur=n,vasc=va,dend=d,bg=b,axon=a)
    for key,updates in (overrides or {}).items():
        if key not in out:
            raise ValueError(f"Unknown tissue parameter group: {key}")
        unknown = set(updates)-set(out[key])
        if unknown:
            raise ValueError(f"Unknown {key} parameters: {sorted(unknown)}")
        for name,value in updates.items():
            if isinstance(value,dict) and isinstance(out[key][name],dict):
                out[key][name].update(value)
            else:
                out[key][name] = value
    for key in out:
        out[key] = {k:np.array(v) if isinstance(v,(tuple,list)) else deepcopy(v) for k,v in out[key].items()}
    return out


def generate_scene(config=None,*,preset="compact",overrides=None,seed=0):
    try:
        import numba
    except ModuleNotFoundError as exc:
        if exc.name != "numba":
            raise
        raise ImportError(
            'Tissue generation requires Numba; install with '
            '`python -m pip install "maomi-optics-preview[tissue]"` '
            'or `python -m pip install ".[tissue]"` from the checkout.'
        ) from exc
    from .tissue_pipeline import simulate_tissue
    from .scene import Scene
    cfg = config or OpticalConfig(volume_um=(32.,32.,20.),depth_um=10.,psf_um=(4.,4.,4.),tail_um=5.)
    params = tissue_parameters(cfg,preset=preset,overrides=overrides)
    neurons,meta = simulate_tissue(**params,seed=seed)
    return Scene(neurons,cfg,vessels=neurons["neur_ves_all"],
                 scan=dict(scan_buff=2,scan_avg=1,sfrac=1,motion=False),
                 metadata=dict(tissue_preset=preset,seed=seed,tissue=meta)).validate()
