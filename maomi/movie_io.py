"""Streaming quantitative TIFF/FITS/NPY and explicitly quantized FFV1 AVI.

Input frames are XY; files use time,Y,X with axes recorded in a sidecar.
A partial file is retained on failure and never advertised as complete.
"""

from pathlib import Path
import json, os, subprocess, time
import numpy as np


def write_movie(frames, path, *, frame_count, shape, field="clean", fps=30.0, intensity_range=None):
    path = Path(path)
    kind = path.suffix.lower()
    partial = path.with_name(path.name + ".partial")
    meta_path = path.with_name(path.name + ".json")
    if any(p.exists() for p in (path, partial, meta_path)):
        raise FileExistsError("Choose a new movie path")
    if (
        int(frame_count) != frame_count
        or frame_count < 1
        or len(shape) != 2
        or any(int(n) != n or n < 1 for n in shape)
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Invalid movie dimensions or frame rate")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(frame_count)
    nx, ny = map(int, shape)
    storage_shape = (count, ny, nx)
    mapping = None
    process = None
    errlog = None
    written = 0
    start = time.perf_counter()
    metadata = dict(
        complete=False,
        format=kind,
        input_axes="XY",
        stored_axes="TYX",
        frames=count,
        shape=list(storage_shape),
        fps=float(fps),
        field=field,
        storage_dtype="float32",
        quantized=False,
    )
    try:
        if kind in (".tif", ".tiff"):
            import tifffile

            mapping = tifffile.memmap(
                partial,
                shape=storage_shape,
                dtype=np.float32,
                photometric="minisblack",
                bigtiff=count * nx * ny * 4 > 3500000000,
                metadata={"axes": "TYX", "fps": fps},
            )
        elif kind == ".npy":
            mapping = np.lib.format.open_memmap(
                partial, mode="w+", dtype=np.float32, shape=storage_shape
            )
        elif kind in (".fits", ".fit"):
            from astropy.io import fits

            header = fits.Header()
            header["SIMPLE"] = True
            header["BITPIX"] = -32
            header["NAXIS"] = 3
            header["NAXIS1"] = nx
            header["NAXIS2"] = ny
            header["NAXIS3"] = count
            header["EXTEND"] = True
            header["CTYPE3"] = "TIME"
            header["CDELT3"] = 1 / fps
            block = header.tostring().encode("ascii")
            size = count * nx * ny * 4
            with partial.open("xb") as f:
                f.write(block)
                f.seek(len(block) + int(np.ceil(size / 2880)) * 2880 - 1)
                f.write(b"\0")
            mapping = np.memmap(
                partial, mode="r+", dtype=">f4", shape=storage_shape, offset=len(block)
            )
        elif kind == ".avi":
            import imageio_ffmpeg

            if intensity_range is None or len(intensity_range) != 2:
                raise ValueError("AVI needs a fixed intensity_range for reversible calibration")
            low, high = map(float, intensity_range)
            if not np.isfinite([low, high]).all() or high <= low:
                raise ValueError("Invalid intensity range")
            metadata.update(
                storage_dtype="uint16",
                quantized=True,
                intensity_range=[low, high],
                quantization_step=(high - low) / 65535,
                codec="FFV1 lossless on uint16 codes",
            )
            errlog = path.with_name(path.name + ".encoder.log").open("xb")
            process = subprocess.Popen(
                [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "rawvideo",
                    "-pixel_format",
                    "gray16le",
                    "-video_size",
                    f"{nx}x{ny}",
                    "-framerate",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-an",
                    "-c:v",
                    "ffv1",
                    "-level",
                    "3",
                    "-pix_fmt",
                    "gray16le",
                    "-f",
                    "avi",
                    str(partial),
                ],
                stdin=subprocess.PIPE,
                stderr=errlog,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        else:
            raise ValueError("Supported streaming formats: .tif, .fits, .npy, .avi")
        for item in frames:
            if written >= count:
                raise ValueError("Iterator produced too many frames")
            a = np.asarray(item[field] if isinstance(item, dict) else item, dtype=np.float32)
            if a.shape != (nx, ny) or not np.isfinite(a).all():
                raise ValueError("Frame has invalid shape or nonfinite values")
            if process is not None:
                if a.min() < low or a.max() > high:
                    raise ValueError("AVI intensity_range would clip data")
                codes = np.rint((a.T.astype(float) - low) * 65535 / (high - low)).astype("<u2")
                process.stdin.write(codes.tobytes(order="C"))
            else:
                mapping[written] = a.T
            written += 1
        if written != count:
            raise ValueError(f"Iterator ended after {written}, expected {count} frames")
        if process is not None:
            process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("FFmpeg failed; encoder log retained")
        if mapping is not None:
            mapping.flush()
            del mapping
            mapping = None
        partial.rename(path)
        metadata.update(
            complete=True,
            written_frames=written,
            seconds=time.perf_counter() - start,
            file_bytes=path.stat().st_size,
        )
    except BaseException as exc:
        metadata.update(written_frames=written, error=str(exc))
        raise
    finally:
        if mapping is not None:
            mapping.flush()
            del mapping
        if process is not None and process.poll() is None:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.terminate()
            process.wait()
        if errlog is not None:
            errlog.close()
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def read_movie(path):
    """Read own export into XYT; AVI calibration comes from its required sidecar."""
    path = Path(path)
    meta = json.loads(path.with_name(path.name + ".json").read_text(encoding="utf-8"))
    if not meta["complete"]:
        raise ValueError("Cannot read incomplete export")
    kind = path.suffix.lower()
    if kind in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(path)
    elif kind == ".npy":
        a = np.load(path)
    elif kind in (".fits", ".fit"):
        from astropy.io import fits

        a = fits.getdata(path).astype(np.float32)
    elif kind == ".avi":
        import imageio_ffmpeg

        raw = subprocess.check_output(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray16le",
                "pipe:1",
            ],
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        a = np.frombuffer(raw, dtype="<u2").reshape(meta["shape"]).astype(float)
        low, high = meta["intensity_range"]
        a = low + a * (high - low) / 65535
    else:
        raise ValueError("Unsupported movie format")
    return np.asarray(a).reshape(meta["shape"]).transpose(2, 1, 0)
