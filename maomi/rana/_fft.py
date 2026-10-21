"""Reusable compact-support local-PSF FFT plan for CPU and CUDA acquisition."""
from collections import OrderedDict, defaultdict
import numpy as np
from scipy import fft
from ._backends import get_backend


class LocalFFTPlan:
    def __init__(self, shape, kernels, weights, *, plane, detection, correlate,
                 device, batch_size=8, cache_mb=256):
        self.shape, self.kernels = tuple(shape), kernels
        self.plane, self.detection, self.correlate = plane, detection, correlate
        self.device, self.batch_size = device, batch_size if device != "cpu" else 1
        self.cuda = device != "cpu"
        self.backend = get_backend(device)
        self.cache, self.cache_bytes = OrderedDict(), 0
        self.cache_limit = int(cache_mb*1024**2)
        self.stats = dict(kernel_fft_builds=0, cache_hits=0)
        self.dims = (1,2) if plane else (1,2,3)
        kshape = kernels.shape[2:]
        centers = np.array([(k-1-k//2) if correlate else k//2 for k in kshape])
        self.groups = defaultdict(list)
        wx, wy = weights
        for i in range(len(wx)):
            for j in range(len(wy)):
                xi, yi = np.flatnonzero(wx[i]), np.flatnonzero(wy[j])
                if not len(xi) or not len(yi):
                    continue
                lo, hi = np.array([xi[0],yi[0]]), np.array([xi[-1]+1,yi[-1]+1])
                if detection:
                    start, end = lo, hi
                    target_start = np.maximum(0, lo-centers[:2])
                    target_end = np.minimum(shape[:2], hi+np.array(kshape[:2])-1-centers[:2])
                    response_start = target_start-lo+centers[:2]
                    w = wx[i,lo[0]:hi[0],None]*wy[j,None,lo[1]:hi[1]]
                else:
                    start = np.maximum(0, lo-np.array(kshape[:2]))
                    end = np.minimum(shape[:2], hi+np.array(kshape[:2]))
                    target_start, target_end = lo, hi
                    response_start = lo-start+centers[:2]
                    w = wx[i,lo[0]:hi[0],None]*wy[j,None,lo[1]:hi[1]]
                source_slice = tuple(slice(int(a),int(b)) for a,b in zip(start,end))
                target = tuple(slice(int(a),int(b)) for a,b in zip(target_start,target_end))
                response = tuple(slice(int(a),int(a+b-c)) for a,b,c in zip(response_start,target_end,target_start))
                if not plane:
                    response += (slice(int(centers[2]),int(centers[2]+shape[2])),)
                source_shape = tuple(end-start)+(shape[2],)
                lengths = tuple(fft.next_fast_len(source_shape[d]+kshape[d]-1) for d in ((0,1) if plane else (0,1,2)))
                self.groups[(source_shape,lengths)].append((i,j,source_slice,target,response,self.array(w)))

    def array(self, a):
        return self.backend.array(a)

    def _kernel_fft(self, entries, lengths):
        key = (tuple((x[0],x[1]) for x in entries),lengths)
        if key in self.cache:
            self.stats["cache_hits"] += 1
            self.cache.move_to_end(key)
            return self.cache[key]
        a = np.stack([self.kernels[i,j] for i,j,*_ in entries])
        if self.correlate:
            a = np.flip(a, axis=(1,2) if self.plane else (1,2,3))
        spec = self.backend.rfftn(self.array(a),lengths,self.dims)
        size = self.backend.nbytes(spec)
        self.stats["kernel_fft_builds"] += 1
        if size <= self.cache_limit:
            while self.cache and self.cache_bytes+size > self.cache_limit:
                _, old = self.cache.popitem(last=False)
                self.cache_bytes -= self.backend.nbytes(old)
            self.cache[key] = spec
            self.cache_bytes += size
        return spec

    def __call__(self, source):
        source = self.array(source)
        out = self.backend.zeros(self.shape[:2] if self.plane else self.shape)
        for (_,lengths), group in self.groups.items():
            # Limit temporary batch spectra to about 128 MiB. A single tile is
            # always retained; the FFT cache has a separate bounded budget.
            per_tile = max(1,int(np.prod(lengths))*64*(self.shape[2] if self.plane else 1))
            batch_size = max(1,min(self.batch_size,128*1024**2//per_tile))
            for start in range(0,len(group),batch_size):
                entries = group[start:start+batch_size]
                parts = [source[sl]*w[...,None] if self.detection else source[sl]
                         for _,_,sl,_,_,w in entries]
                tiles = parts[0][None] if len(parts)==1 else self.backend.stack(parts)
                k = self._kernel_fft(entries,lengths)
                spec = self.backend.rfftn(tiles,lengths,self.dims)*k
                if self.plane:
                    spec = self.backend.sum(spec,axis=3)
                full = self.backend.irfftn(spec,lengths,self.dims)
                for n,(_,_,_,target,response,w) in enumerate(entries):
                    piece = full[(n,)+response]
                    out[target] += piece if self.detection else piece*(w if self.plane else w[...,None])
        return out

    def clear(self):
        self.cache.clear()
        self.cache_bytes = 0
