"""Portable, chunked scene/optics archives. Numeric data only; never pickle."""
from pathlib import Path
import json
from dataclasses import asdict
import numpy as np
from .mat_io import _h5
from .scene import Scene
from .optics import OpticalConfig


def jsonable(x):
    if isinstance(x,np.ndarray):
        return x.tolist()
    if isinstance(x,np.generic):
        return x.item()
    if isinstance(x,dict):
        return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [jsonable(v) for v in x]
    if isinstance(x,Path):
        return str(x)
    return x


class StoredArray:
    def __init__(self,path,key):
        self.path,self.key = str(Path(path).resolve()),key
        with _h5().File(self.path,"r") as f:
            self.shape,self.dtype = f[key].shape,f[key].dtype
        self.ndim = len(self.shape)

    def __getitem__(self,key):
        with _h5().File(self.path,"r") as f:
            return f[self.key][key]

    def __array__(self,dtype=None,copy=None):
        a = self[()]
        return np.array(a,dtype=dtype,copy=True) if copy else np.asarray(a,dtype=dtype)


class StoredComponents:
    def __init__(self,path,key):
        self.path,self.key = str(Path(path).resolve()),key
        with _h5().File(self.path,"r") as f:
            self.count = len(f[key]["offsets"])-1
            self.columns = int(f[key].attrs["columns"])

    def __len__(self):
        return self.count

    def __iter__(self):
        with _h5().File(self.path,"r") as f:
            g = f[self.key]
            offsets = g["offsets"][()]
            for a,b in zip(offsets[:-1],offsets[1:]):
                yield tuple(g[str(c)][int(a):int(b)] for c in range(self.columns))

    def __getitem__(self,i):
        if not 0 <= i < self.count:
            raise IndexError(i)
        with _h5().File(self.path,"r") as f:
            g = f[self.key]
            a,b = g["offsets"][i:i+2]
            return tuple(g[str(c)][int(a):int(b)] for c in range(self.columns))


def _array(group,key,value):
    shape = tuple(value.shape)
    dtype = value.dtype if hasattr(value,"dtype") else np.asarray(value).dtype
    ds = group.create_dataset(key,shape=shape,dtype=dtype,chunks=True if shape and all(shape) else None)
    if not shape or not all(shape):
        ds[()] = np.asarray(value)
    else:
        # Bound transfer buffers, even for a very large lazy volume or activity.
        slab = max(1,min(shape[0],8*1024**2//max(1,int(np.prod(shape[1:]))*np.dtype(dtype).itemsize)))
        for start in range(0,shape[0],slab):
            ds[start:start+slab] = value[start:start+slab]


def _components(group,key,rows,columns):
    g = group.create_group(key)
    g.attrs["columns"] = columns
    datasets = [g.create_dataset(str(c),shape=(0,),maxshape=(None,),dtype=np.int64 if c==0 else bool if c==2 else np.float64,chunks=True)
                for c in range(columns)]
    offsets = [0]
    # Append once per component; neither rows nor their full concatenation are held.
    for row in rows:
        n = np.asarray(row[0]).size
        end = offsets[-1]+n
        for c,ds in enumerate(datasets):
            a = np.asarray(row[c]).ravel()
            if a.size == 1 and n != 1:
                a = np.full(n,a[0])
            if a.size != n:
                raise ValueError("Sparse archive row has mismatched values")
            ds.resize((end,))
            ds[offsets[-1]:end] = a
        offsets.append(end)
    g.create_dataset("offsets",data=np.asarray(offsets,np.int64))


def save_scene(scene,path):
    """Save dense arrays in bounded slabs and sparse rows in packed form."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    partial = path.with_name(path.name+".partial")
    with _h5().File(partial,"w") as f:
        f.attrs["schema"] = "maomi.scene.v1"
        f.attrs["metadata"] = json.dumps(jsonable(dict(config=asdict(scene.config),scan=scene.scan,
            noise=scene.noise,spikes=scene.spikes,tpm=scene.tpm,metadata=scene.metadata)),allow_nan=False)
        neurons = f.create_group("neurons")
        for k,v in scene.neurons.items():
            if k in ("gp_vals","gp_nuc","bg_proc"):
                _components(neurons,k,v,3 if k=="gp_vals" else 2)
            elif hasattr(v,"shape") and np.dtype(v.dtype).kind != "O":
                _array(neurons,k,v)
        if scene.vessels is not None:
            _array(f,"vessels",scene.vessels)
        if scene.activity is not None:
            a = f.create_group("activity")
            for k,v in scene.activity.items():
                _array(a,k,v if hasattr(v,"shape") else np.atleast_2d(v))
    partial.replace(path)
    return path


def load_scene(path):
    path = Path(path)
    with _h5().File(path,"r") as f:
        if f.attrs["schema"] != "maomi.scene.v1":
            raise ValueError("Unsupported scene archive")
        meta = json.loads(f.attrs["metadata"])
        neurons = {k:StoredComponents(path,"neurons/"+k) if isinstance(v,_h5().Group)
                   else StoredArray(path,"neurons/"+k) for k,v in f["neurons"].items()}
        activity = {k:StoredArray(path,"activity/"+k) for k in f["activity"]} if "activity" in f else None
        vessels = StoredArray(path,"vessels") if "vessels" in f else None
    return Scene(neurons,OpticalConfig(**meta.pop("config")),activity,vessels,**meta).validate()


def _write_tree(group,value):
    for k,v in value.items():
        if isinstance(v,dict):
            _write_tree(group.create_group(k),v)
        elif isinstance(v,np.ndarray):
            _array(group,k,v)
        elif v is not None:
            group.attrs[k] = json.dumps(jsonable(v),allow_nan=False)


def _read_tree(group):
    out = {k:json.loads(v) for k,v in group.attrs.items()}
    out.update({k:_read_tree(v) if isinstance(v,_h5().Group) else v[()] for k,v in group.items()})
    return out


def save_optics(optics,path):
    """Persist reusable propagation and local PSF fields, without device caches."""
    from .rana_depth import DepthPSFBank
    def pack(value):
        out = {k:v for k,v in value.items() if k not in ("rana","channels")}
        if "channels" in value:
            out["channels"] = {k:pack(v) for k,v in value["channels"].items()}
        if value.get("rana") is not None:
            op = value["rana"]
            out["_rana_archive"] = dict(native=op.native,origin_um=list(op.origin_um),
                z_origin_um=op.z_origin_um,depths_um=op.bank.depths_um if op.bank else np.array([]),
                fields={str(i):dict(kernels=f.kernels,x_um=f.x_um,y_um=f.y_um,
                    voxel_size_um=f.voxel_size_um,wavelength_nm=f.wavelength_nm,
                    operator=f.operator,provenance=f.provenance) for i,f in enumerate(op.fields)})
        return out
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    partial = path.with_name(path.name+".partial")
    with _h5().File(partial,"w") as f:
        f.attrs["_schema"] = json.dumps("maomi.optics.v1")
        _write_tree(f,pack(optics))
    partial.replace(path)


def load_optics(path,*,device="cpu"):
    from .rana_addon import RanaOperator
    from .enhanced import FullFieldPSF
    from .rana_depth import DepthPSFBank
    with _h5().File(path,"r") as f:
        out = _read_tree(f)
    if out.pop("_schema") != "maomi.optics.v1":
        raise ValueError("Unsupported optics archive")
    def unpack(value):
        if "channels" in value:
            value["channels"] = {k:unpack(v) for k,v in value["channels"].items()}
        if "_rana_archive" in value:
            d = value.pop("_rana_archive")
            fields = tuple(FullFieldPSF(**d["fields"][str(i)]) for i in range(len(d["fields"])))
            field = DepthPSFBank(d["depths_um"],fields) if len(d["depths_um"]) else fields[0]
            value["rana"] = RanaOperator(field,native=d["native"],origin_um=d["origin_um"],
                                         z_origin_um=d["z_origin_um"],device=device)
        return value
    return unpack(out)
