"""Bounded activity reads and reusable sparse buffers with NAOMi write order."""
import numpy as np


class ActivitySource:
    def __init__(self, activity, counts, *, chunk_frames=256):
        self.arrays = []
        self.chunk_frames = int(chunk_frames)
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        soma = activity["soma"]
        if not hasattr(soma,"shape") or len(soma.shape) != 2:
            soma = np.atleast_2d(soma)
        self.frames = soma.shape[1]
        if self.frames < 1:
            raise ValueError("At least one activity frame required")
        for k, default, count in zip(("soma","dend","bg"),(soma,soma,None),counts):
            a = activity.get(k,default)
            if a is not None:
                if not hasattr(a,"shape") or len(a.shape) != 2:
                    a = np.atleast_2d(a)
                if tuple(a.shape) != (count,self.frames):
                    raise ValueError(f"{k}: component/activity counts or frame counts differ")
            self.arrays.append(a)
        self.counts = counts
        self.minima = [np.full(n,np.inf,np.float32) for n in counts]
        for start in range(0,self.frames,self.chunk_frames):
            for k,a in enumerate(self._read(start)):
                self.minima[k] = np.minimum(self.minima[k],a.min(axis=1))
        self._start, self._block = -1, None

    def _read(self,start):
        stop = min(self.frames,start+self.chunk_frames)
        out = []
        for a,n in zip(self.arrays,self.counts):
            v = np.zeros((n,stop-start),np.float32) if a is None else np.asarray(a[:,start:stop],np.float32)
            if not np.isfinite(v).all() or (v < 0).any():
                raise ValueError("Activity must be finite nonnegative fluorescence")
            out.append(v)
        return out

    def frame(self,t,nuclear):
        start = t//self.chunk_frames*self.chunk_frames
        if start != self._start:
            self._start, self._block = start,self._read(start)
        columns = [v[:,t-start] for v in self._block]
        if nuclear:
            nuc = columns[0].copy()
            nuc[nuc < .01] = 0
            return [np.zeros_like(columns[0]),nuc,np.zeros_like(columns[1]),np.zeros_like(columns[2])]
        delta = [v-m for v,m in zip(columns,self.minima)]
        for v in delta:
            v[v < .01] = 0
        return [delta[0],np.empty(0,np.float32),delta[1],delta[2]]

    def at_time(self, frame_position):
        """Raw activity interpolated at a fractional frame, with endpoint hold.

        Timed acquisition has no activity cutoff or minimum subtraction.
        Reads stay bounded by chunk_frames, including across chunk boundaries.
        """
        if not np.isfinite(frame_position):
            raise ValueError("Activity time must be finite")
        position = float(np.clip(frame_position, 0, self.frames-1))
        lo = int(np.floor(position))
        hi = min(lo+1, self.frames-1)
        def column(t):
            start = t//self.chunk_frames*self.chunk_frames
            if start != self._start:
                self._start, self._block = start, self._read(start)
            return [v[:,t-start] for v in self._block]
        a, b = column(lo), column(hi)
        return [x+(position-lo)*(y-x) for x,y in zip(a,b)]


def _fill(buffer, indices, values, offsets, additive, amplitudes):
    for j in range(len(amplitudes)):
        amp = amplitudes[j]
        if amp > 0:
            for p in range(offsets[j],offsets[j+1]):
                if additive[j]:
                    buffer[indices[p]] += values[p]*amp
                else:
                    buffer[indices[p]] = values[p]*amp


_compiled = None


class SparseMovieBuffer:
    def __init__(self,shape,groups,*,backend="auto"):
        self.shape = tuple(shape)
        self.buffer = np.zeros(np.prod(shape),np.float32)
        ids, values, offsets, add = [],[],[0],[]
        for rows, additive in groups:
            for i,v in rows:
                ids.append(np.asarray(i,np.int64))
                values.append(np.asarray(v,np.float32))
                offsets.append(offsets[-1]+len(i))
                add.append(additive)
        self.ids = np.concatenate(ids) if ids else np.empty(0,np.int64)
        self.values = np.concatenate(values) if values else np.empty(0,np.float32)
        self.offsets, self.add = np.array(offsets,np.int64),np.array(add,bool)
        self.touched = np.unique(self.ids)
        if backend not in ("auto","numpy","numba"):
            raise ValueError("sparse_backend must be auto, numpy or numba")
        global _compiled
        self.fill = None
        if backend != "numpy":
            try:
                from numba import njit
                if _compiled is None:
                    _compiled = njit(cache=True)(_fill)
                self.fill = _compiled
            except ImportError:
                if backend == "numba":
                    raise ImportError('Install the tissue extra for the compiled sparse writer')
        self.backend = "numba" if self.fill is not None else "numpy"

    def frame(self, amplitudes):
        self.buffer[self.touched] = 0
        a = np.concatenate(amplitudes).astype(np.float32,copy=False)
        if len(a) != len(self.add):
            raise ValueError("Sparse amplitudes differ from component count")
        if self.fill is not None:
            self.fill(self.buffer,self.ids,self.values,self.offsets,self.add,a)
        else:
            for j in np.flatnonzero(a > 0):
                sl = slice(self.offsets[j],self.offsets[j+1])
                if self.add[j]:
                    np.add.at(self.buffer,self.ids[sl],self.values[sl]*a[j])
                else:
                    self.buffer[self.ids[sl]] = self.values[sl]*a[j]
        return self.buffer.reshape(self.shape,order="F")
