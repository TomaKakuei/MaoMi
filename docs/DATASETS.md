# NAOMi datasets

Original data: **[OSF project 863j9](https://osf.io/863j9/)** and its
**[file browser](https://osf.io/863j9/files/)**.
The [NAOMi paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8217135/) identifies
this repository in its data-availability statement. Please cite Song et al.
(2021) when using these data.

## Choose a dataset

| OSF folder | Contents |
| --- | --- |
| `NAOMi_Figure2` | `Fig2Sim` and `Fig2Sim20K`; 16-bit and 32-bit TIFF folders |
| `NAOMi_Figure3` | Long movies, ideal traces, estimates, analysis code and `Figure3_largeVolume.mat` |
| `NAOMi_Figure4` | Sparse labeling, nuclear labeling and volumetric imaging |
| `NAOMi_SubsampledVideo` | Different spatial sampling fractions; 16-bit and 32-bit TIFF |
| `NAOMI_extraMovies` | Depth, activity, labeling, power, Bessel-beam and misalignment variants |
| `NeuropilData` | `neuropilTMP_20200416_realData.mat` |

Names and links were checked against the public OSF API on 19 October 2026.
Movies are split into numbered chunks; retain their ordering. Choose 32-bit
data for quantitative comparisons when available.

## Download and read a movie

- [Figure 3 movie, chunk 00001](https://osf.io/download/na874/):
  `movie_standard_standard_long_00001.tif`, 2,521,700,007 bytes.
- [Subsampled sfrac02, chunk 00026](https://osf.io/download/zurcs/):
  `longMovSampled20190524_sfrac02_00026.tif`, 151,299,307 bytes.
  This final short chunk is convenient for a smaller loading example.

```sh
python -m pip install ".[io]"
curl -L --fail --output sample.tif https://osf.io/download/zurcs/
```

On Windows PowerShell, use `curl.exe`. Read individual frames without loading
the whole movie:

```python
import tifffile

with tifffile.TiffFile("sample.tif") as movie:
    print("frames:", len(movie.pages))
    frame_yx = movie.pages[0].asarray()
    print(frame_yx.shape, frame_yx.dtype)
    frame_xy = frame_yx.T  # MaoMi frame axes are XY.
```

A TIFF is an already formed observation. Changing its source optics requires
the corresponding anatomy, component activity and acquisition parameters.
Reblurring a TIFF is not a new forward simulation of the original tissue.

## Reuse MATLAB source data

[Figure3_largeVolume.mat](https://osf.io/download/b4jky/) is listed at
4,042,087,963 bytes. Inspect it without loading the volume, then use the
public importer:

```sh
python -m pip install ".[mat]"
python -m maomi inspect Figure3_largeVolume.mat
```

```python
from maomi import load_naomi_mat

scene = load_naomi_mat(
    "Figure3_largeVolume.mat",
    roi=((400, 432), (400, 432), (60, 80)),
    optical_overrides={"psf_um": (4, 4, 4), "tail_um": 5},
)
# source anatomy is retained at native sampling; no resize or tiling.
scene.generate_activity(frames=30, seed=19)  # explicitly synthetic activity
```

This example keeps a small native-voxel region and explicitly chooses a focal
kernel/tail that fits the stored vessel extent. Physical depth comes from the
source. v5-v7.2 and v7.3 are supported; v7.3 arrays/cell rows are lazy by default.
One-based MATLAB sparse indices are remapped to the ROI while component IDs
remain aligned with activity. Missing traces are not fabricated.

Use `activity_path` / `activity_key` to import recorded simulator traces;
derived analysis MAT files are not automatically source anatomy. Native RANA
requires local propagated PSFs or a supplied complete XYZ field; a mean PSF
does not contain these missing local shapes.

See [the complete workflow](WORKFLOWS.md) for CPU/CUDA acquisition, RANA,
source parameter mapping, optical padding, checkpoints and training-data access.
Some original source volumes were available by request, as the paper explains.
