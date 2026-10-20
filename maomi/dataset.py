"""Atomic movie chunks, resumable progress, and a lazy training-data reader."""
from pathlib import Path
import json
import hashlib
import time
import numpy as np


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024**2),b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path,data):
    temp = path.with_name(path.name+".partial")
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding="utf8")
    temp.replace(path)


def write_chunks(stream,path,*,frames,chunk_frames=64,start=0,max_frames=None):
    path = Path(path)
    manifest = path/"dataset.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else dict(
        schema="maomi.dataset.v1",axes="TYX",frames_expected=frames,chunks=[],complete=False)
    if frames != data["frames_expected"] or sum(c["frames"] for c in data["chunks"]) != start:
        raise ValueError("Dataset progress does not match the scanner")
    if max_frames is not None and (int(max_frames)!=max_frames or max_frames < 1):
        raise ValueError("max_frames must be a positive integer")
    count = 0
    batch = []
    started = time.perf_counter()
    def flush():
        if not batch:
            return
        first = batch[0]["index"]
        name = f"frames_{first:08d}.npz"
        target = path/name
        # A chunk can exist after interruption between rename and manifest commit.
        # It is safely recomputed from archived inputs and per-frame noise seeds.
        temp = target.with_name(target.name+".partial")
        arrays = {"clean":np.stack([b["clean"].T for b in batch]),
                  "index":np.array([b["index"] for b in batch])}
        if batch[0]["noisy"] is not None:
            arrays["noisy"] = np.stack([b["noisy"].T for b in batch])
        with temp.open("wb") as f:
            np.savez(f,**arrays)
        digest = _sha256(temp)
        temp.replace(target)
        data["chunks"].append(dict(file=name,start=first,frames=len(batch),sha256=digest))
        data["complete"] = first+len(batch) == frames
        data["write_seconds"] = data.get("write_seconds",0.)+time.perf_counter()-flush.last
        flush.last = time.perf_counter()
        _atomic_json(manifest,data)
        batch.clear()
    flush.last = started
    for frame in stream:
        if frame["index"] != start+count:
            raise ValueError("Noncontiguous scanner frame")
        batch.append(frame)
        count += 1
        if len(batch) >= chunk_frames:
            flush()
        if max_frames is not None and count >= max_frames:
            break
    flush()
    if not manifest.exists():
        _atomic_json(manifest,data)
    return data


class MovieDataset:
    """Lazy frame pairs in YX order; compatible with torch DataLoader."""
    def __init__(self,path,*,allow_incomplete=False,verify=False):
        self.path = Path(path)
        mf = self.path/"dataset.json"
        self.manifest = json.loads(mf.read_text()) if mf.exists() else dict(chunks=[],complete=False)
        if self.manifest.get("schema","maomi.dataset.v1") != "maomi.dataset.v1":
            raise ValueError("Unsupported dataset schema")
        if not self.manifest["complete"] and not allow_incomplete:
            raise ValueError("Dataset is incomplete")
        cursor = 0
        for c in self.manifest["chunks"]:
            p = self.path/c["file"]
            if p.parent.resolve() != self.path.resolve() or c["start"] != cursor or not p.is_file():
                raise ValueError("Invalid or missing dataset chunk")
            if verify and _sha256(p)!=c["sha256"]:
                raise ValueError("Dataset chunk checksum mismatch")
            cursor += c["frames"]
        self.frames,self.complete = cursor,self.manifest["complete"]
        if self.complete and cursor != self.manifest.get("frames_expected"):
            raise ValueError("Complete dataset has an inconsistent frame count")
        self._cached = None

    def __len__(self):
        return self.frames

    def __getitem__(self,i):
        if i < 0:
            i += self.frames
        if not 0 <= i < self.frames:
            raise IndexError(i)
        c = next(c for c in self.manifest["chunks"] if c["start"] <= i < c["start"]+c["frames"])
        if self._cached is None or self._cached[0] != c["file"]:
            with np.load(self.path/c["file"],allow_pickle=False) as z:
                self._cached = (c["file"],{k:z[k] for k in ("clean","noisy") if k in z})
        j = i-c["start"]
        return {**{k:v[j].copy() for k,v in self._cached[1].items()},"index":i}

    def scene(self):
        from .archive import load_scene
        return load_scene(self.path/"scene.h5")

    @property
    def metadata(self):
        return json.loads((self.path/"run.json").read_text(encoding="utf8"))
