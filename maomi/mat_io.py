"""NAOMi MAT v5-v7.2/v7.3 import, with lazy HDF5 arrays and sparse ROI remapping.

No MATLAB runtime is required. Public arrays are XYZ and component indices
remain one-based, column-major. Crop bounds are zero-based, half-open XYZ.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from .scene import Scene
from .optics import OpticalConfig


def _h5():
    try:
        import h5py
    except ImportError as exc:
        raise ImportError('MAT v7.3 requires pip install "maomi-optics-preview[mat]"') from exc
    return h5py


def _kind(node):
    value = node.attrs.get("MATLAB_class", b"")
    return value.decode() if isinstance(value, bytes) else str(value)


def _decode(node):
    h5py = _h5()
    if isinstance(node, h5py.Group):
        if "MATLAB_sparse" in node.attrs:
            from scipy.sparse import csc_matrix
            return csc_matrix((node["data"][()], node["ir"][()], node["jc"][()]),
                              shape=(int(node.attrs["MATLAB_sparse"]), len(node["jc"])-1))
        return {k: _decode(v) for k, v in node.items() if not k.startswith("#")}
    a = node[()]
    if node.attrs.get("MATLAB_empty", 0):
        return np.empty(tuple(np.asarray(a, int).ravel()))
    if h5py.check_dtype(ref=node.dtype) is not None:
        out = np.empty(a.shape, object)
        for idx in np.ndindex(a.shape):
            out[idx] = _decode(node.file[a[idx]]) if a[idx] else np.array([])
        # Cells retain their row/column structure, including singleton rows.
        return out.T
    if a.dtype.names and {"real", "imag"} <= set(a.dtype.names):
        a = a["real"] + 1j*a["imag"]
    a = a.T
    if _kind(node) == "char":
        return "".join(chr(int(x)) for x in a.ravel() if x)
    if _kind(node) == "logical":
        a = a.astype(bool)
    return a.item() if a.size == 1 else a


class MatArray:
    """Reopen-on-read HDF5 numeric view; no open file handle escapes a call."""
    def __init__(self, path, key, roi=None):
        self.path, self.key = str(Path(path).resolve()), key
        with _h5().File(self.path, "r") as f:
            node = f[key]
            original = tuple(reversed(node.shape))
            self.dtype = np.dtype(bool) if _kind(node) == "logical" else node.dtype
        self.roi = tuple((0, n) for n in original) if roi is None else tuple(map(tuple, roi))
        if len(self.roi) != len(original) or any(a < 0 or b > n or a >= b for (a,b),n in zip(self.roi,original)):
            raise ValueError(f"ROI outside {key}: {original}")
        self.shape = tuple(b-a for a,b in self.roi)
        self.ndim = len(self.shape)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        if any(k is Ellipsis for k in key):
            i = next(i for i,k in enumerate(key) if k is Ellipsis)
            key = key[:i] + (slice(None),)*(self.ndim-len(key)+1) + key[i+1:]
        if len(key) > self.ndim:
            raise IndexError("Too many indices for MAT array")
        key += (slice(None),)*(self.ndim-len(key))
        selections, squeeze = [], []
        for d, (k, (a,b)) in enumerate(zip(key, self.roi)):
            if isinstance(k, (int, np.integer)):
                k = int(k)
                k = k + b-a if k < 0 else k
                if not 0 <= k < b-a:
                    raise IndexError(k)
                selections.append(slice(a+k, a+k+1))
                squeeze.append(d)
            elif isinstance(k, slice):
                start, stop, step = k.indices(b-a)
                if step <= 0:
                    raise ValueError("Lazy MAT views require positive slice steps")
                selections.append(slice(a+start, a+stop, step))
            else:
                raise TypeError("Lazy MAT indexing supports integers and slices")
        with _h5().File(self.path, "r") as f:
            out = f[self.key][tuple(reversed(selections))].T.astype(self.dtype, copy=False)
        return np.squeeze(out, axis=tuple(squeeze)) if squeeze else out

    def __array__(self, dtype=None, copy=None):
        a = self[tuple(slice(None) for _ in self.shape)]
        return np.array(a, dtype=dtype, copy=True) if copy else np.asarray(a, dtype=dtype)


def _crop_row(row, old_shape, roi):
    ids = np.asarray(row[0]).ravel()
    if not np.equal(ids, np.floor(ids)).all() or np.any(ids < 1) or np.any(ids > np.prod(old_shape)):
        raise ValueError("Invalid MATLAB one-based component indices")
    xyz = np.array(np.unravel_index(ids.astype(np.int64)-1, old_shape, order="F"))
    lower, upper = np.asarray(roi).T
    keep = np.all((xyz >= lower[:,None]) & (xyz < upper[:,None]), axis=0)
    new_ids = np.ravel_multi_index(xyz[:,keep]-lower[:,None], upper-lower, order="F")+1
    values = []
    for item in row[1:]:
        a = np.asarray(item).ravel()
        if a.size == 1:
            a = np.full(len(ids), a[0], dtype=a.dtype)
        if a.size != len(ids):
            raise ValueError("Sparse component values must match their indices")
        values.append(a[keep])
    return (new_ids, *values)


class MatCells:
    """Lazy NAOMi rows; crop preserves component count and activity ordering."""
    def __init__(self, path, key, columns, old_shape=None, roi=None):
        self.path, self.key, self.columns = str(Path(path).resolve()), key, columns
        self.old_shape, self.roi = old_shape, roi
        with _h5().File(self.path, "r") as f:
            node = f[key]
            shape = node.shape
            if node.attrs.get("MATLAB_empty",0) or np.prod(shape)==0:
                self.count = 0
                return
        if len(shape) != 2 or shape[0] != columns:
            raise ValueError(f"{key} must be a MATLAB N-by-{columns} cell array")
        self.count = shape[1]

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.count == 0:
            return
        with _h5().File(self.path, "r") as f:
            refs = f[self.key]
            for i in range(self.count):
                row = tuple(_decode(f[r]) if r else np.array([]) for r in refs[:,i])
                yield _crop_row(row, self.old_shape, self.roi) if self.roi is not None else row

    def __getitem__(self, i):
        if not 0 <= i < self.count:
            raise IndexError(i)
        with _h5().File(self.path, "r") as f:
            row = tuple(_decode(f[r]) if r else np.array([]) for r in f[self.key][:,i])
        return _crop_row(row, self.old_shape, self.roi) if self.roi is not None else row


def _rows(value, columns):
    a = np.asarray(value, object)
    if a.size == 0:
        return []
    if a.ndim == 1 and len(a) == columns:
        return [tuple(a)]
    if a.ndim != 2 or a.shape[1] != columns:
        raise ValueError(f"Expected N-by-{columns} MATLAB cell rows, got {a.shape}")
    return [tuple(row) for row in a]


def _scalar_struct(value):
    if isinstance(value, dict):
        return value
    if hasattr(value, "_fieldnames"):
        return {k: _scalar_struct(getattr(value,k)) if hasattr(getattr(value,k), "_fieldnames")
                else getattr(value,k) for k in value._fieldnames}
    raise ValueError("Expected a scalar MATLAB struct")


def _is_hdf(path):
    # MATLAB v7.3 usually places the HDF5 signature after a 512-byte user block.
    with open(path, "rb") as stream:
        head = stream.read(520)
    return head.startswith(b"\x89HDF") or head[512:520] == b"\x89HDF\r\n\x1a\n"


def inspect_naomi_mat(path):
    """Inspect top-level variables without loading tissue arrays."""
    if _is_hdf(path):
        with _h5().File(path, "r") as f:
            return {k: {"kind": _kind(v), "shape": list(reversed(v.shape)) if hasattr(v,"shape") else None}
                    for k,v in f.items() if not k.startswith("#")}
    from scipy.io import whosmat
    return {k: {"shape": list(shape), "kind": kind} for k,shape,kind in whosmat(path)}


def load_naomi_mat(path, *, roi=None, lazy=True, tissue_key="vol_out",
                   activity_key=None, activity_path=None, optical_overrides=None):
    """Load source tissue into Scene. Missing activity stays None.

    roi=((x0,x1),(y0,y1),(z0,z1)) uses source voxel indices. All components
    (including empty cropped ones) retain their original IDs. v5 MAT arrays
    are decoded eagerly by SciPy; v7.3 dense arrays and cells can remain lazy.
    """
    path = Path(path).resolve()
    hdf = _is_hdf(path)
    if hdf:
        with _h5().File(path, "r") as f:
            names = set(f)
            if tissue_key not in f or "neur_vol" not in f[tissue_key]:
                raise ValueError(f"{tissue_key}/neur_vol missing; this may be an analysis MAT, not source tissue")
            vp = _decode(f["vol_params"]) if "vol_params" in f else {}
            pp = _decode(f["psf_params"]) if "psf_params" in f else {}
            keys = set(f[tissue_key])
            source_shape = tuple(reversed(f[tissue_key+"/neur_vol"].shape))
            small = {k: _decode(f[k]) for k in ("scan_params","noise_params","spike_opts","tpm_params") if k in f}
    else:
        raw = loadmat(path, struct_as_record=False, squeeze_me=False)
        names = set(raw)
        def struct(k):
            return _scalar_struct(raw[k].item()) if k in raw else {}
        tissue = struct(tissue_key)
        if "neur_vol" not in tissue:
            raise ValueError(f"{tissue_key}.neur_vol missing; not a source tissue MAT")
        vp, pp = struct("vol_params"), struct("psf_params")
        keys, source_shape = set(tissue), tissue["neur_vol"].shape
        small = {k: struct(k) for k in ("scan_params","noise_params","spike_opts","tpm_params") if k in raw}
        vp = {k: np.asarray(v).squeeze().item() if np.asarray(v).size == 1 else np.asarray(v).squeeze() for k,v in vp.items()}
        pp = {k: np.asarray(v).squeeze().item() if np.asarray(v).size == 1 else np.asarray(v).squeeze() for k,v in pp.items()}
    if len(source_shape) != 3 or "vres" not in vp:
        raise ValueError("XYZ neur_vol and vol_params.vres are required; physical units cannot be guessed")
    r = float(vp["vres"])
    if not np.isfinite(r) or r <= 0:
        raise ValueError("vres must be positive voxels per micrometre")
    crop = tuple((0,n) for n in source_shape) if roi is None else tuple(tuple(x) for x in roi)
    if len(crop) != 3 or any(a != int(a) or b != int(b) or a < 0 or b > n or a >= b for (a,b),n in zip(crop,source_shape)):
        raise ValueError("ROI must contain three valid integer half-open intervals")
    crop = tuple((int(a),int(b)) for a,b in crop)
    shape = tuple(b-a for a,b in crop)
    kwargs = dict(volume_um=tuple(n/r for n in shape), vres=r,
                  depth_um=float(vp.get("vol_depth", 0))+crop[2][0]/r)
    mapping = dict(NA="NA", objNA="objNA", n="n", wavelength_um="lambda",
                   focal_length_mm="obj_fl", ss="ss", psf_um="psf_sz", prop_um="prop_sz",
                   tail_um="taillength", n_diff="n_diff", hemoabs="hemoabs", propcrop="propcrop",
                   scaling="scaling", temporal_length_um="length",
                   scatter_scales_um="scatter_sz", scatter_weights="scatter_wt",
                   zernike_weights="zernikeWt")
    for dest, src in mapping.items():
        if src in pp:
            v = np.asarray(pp[src]).squeeze()
            kwargs[dest] = tuple(v.tolist()) if v.ndim else v.item()
    if "zernike_weights" in kwargs:
        kwargs["zernike_weights"] = tuple(np.atleast_1d(kwargs["zernike_weights"]))
    kwargs.update(optical_overrides or {})
    cfg = OpticalConfig(**kwargs)
    neurons = {}
    if hdf:
        neurons["neur_vol"] = MatArray(path, tissue_key+"/neur_vol", crop)
    else:
        neurons["neur_vol"] = tissue["neur_vol"][tuple(slice(a,b) for a,b in crop)]
    for key, cols in (("gp_vals",3),("gp_nuc",2),("bg_proc",2)):
        if key not in keys:
            continue
        if hdf:
            neurons[key] = MatCells(path, tissue_key+"/"+key, cols, source_shape, crop)
        else:
            neurons[key] = [_crop_row(row, source_shape, crop) for row in _rows(tissue[key],cols)]
    vessels = None
    if "neur_ves_all" in keys:
        if hdf:
            original = MatArray(path,tissue_key+"/neur_ves_all")
            vs = original.shape
        else:
            original = tissue["neur_ves_all"]
            vs = original.shape
        target = np.rint(np.asarray(cfg.parameters()[0].vasc_sz)*r).astype(int)
        required_z = round((cfg.depth_um+cfg.volume_um[2]/2+cfg.psf_um[2]/2)*r)+round(cfg.tail_um*r)
        target[2] = min(vs[2],max(target[2],required_z))
        starts = [(vs[d]-source_shape[d])//2+crop[d][0]+(shape[d]-target[d])//2 for d in (0,1)] + [0]
        bounds = tuple((int(a),int(a+n)) for a,n in zip(starts,target))
        if any(a < 0 or b > n for (a,b),n in zip(bounds,vs)):
            raise ValueError("Stored vessels do not cover the requested optical padding; supply a compatible OpticalConfig")
        vessels = MatArray(path,tissue_key+"/neur_ves_all",bounds) if hdf else original[tuple(slice(a,b) for a,b in bounds)]
    activity = None
    ap = Path(activity_path).resolve() if activity_path else path
    anames = set(inspect_naomi_mat(ap)) if ap != path else names
    ak = activity_key or next((k for k in ("activity","neur_act") if k in anames), None)
    if ak is not None:
        if _is_hdf(ap):
            with _h5().File(ap,"r") as f:
                activity = {k: MatArray(ap,ak+"/"+k) for k in ("soma","dend","bg") if k in f[ak]}
        else:
            ar = loadmat(ap, struct_as_record=False, squeeze_me=False)
            activity = _scalar_struct(ar[ak].item())
    if not lazy:
        neurons = {k: np.asarray(v) if isinstance(v,MatArray) else list(v) if isinstance(v,MatCells) else v for k,v in neurons.items()}
        vessels = None if vessels is None else np.asarray(vessels)
        activity = None if activity is None else {k:np.asarray(v) for k,v in activity.items()}
    def params(key):
        return {k: np.asarray(v).item() if np.asarray(v).size == 1 else v for k,v in small.get(key,{}).items()}
    return Scene(neurons,cfg,activity,vessels,params("scan_params"),params("noise_params"),
                 params("spike_opts"),params("tpm_params"),dict(
                     source=str(path), format="MAT-v7.3" if hdf else "MAT-v5-v7.2",
                     source_shape_xyz=list(source_shape), roi_xyz=crop,
                     source_origin_um=[a/r for a,b in crop], component_ids="preserved",
                     original_psf_parameters=pp,
                     required_optical_overrides=(
                         (["input_pupil"] if str(pp.get("type","gaussian")).strip()!="gaussian" else [])+
                         (["custom_phase or input_pupil"] if "zernikeDst" in pp and np.asarray(pp["zernikeDst"]).size else [])),
                     optical_parameters_defaulted=[k for k,v in mapping.items() if v not in pp])).validate()
