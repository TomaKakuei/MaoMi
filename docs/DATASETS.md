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
4,042,087,963 bytes. Derived analysis MAT files contain different variables
from simulator input files. Inspect the actual file before selecting inputs:

```python
from scipy.io import whosmat, loadmat

print(whosmat("input.mat"))  # MATLAB v5-v7.2
data = loadmat("input.mat", simplify_cells=True)
print([key for key in data if not key.startswith("__")])
```

For v7.3 use `h5py` (`pip install ".[reproduce]"`) to inspect datasets and
attributes. Decode MATLAB struct/cell references and stored axis order before
passing arrays to MaoMi; a raw HDF5 array conversion is not a struct loader.

| Scanner argument | Expected representation |
| --- | --- |
| `neurons` | `neur_vol` XYZ; `gp_vals` rows `(indices, values, is_soma)`; optional `gp_nuc` / `bg_proc` |
| Sparse indices | **One-based MATLAB column-major**; do not subtract one before calling the scanner |
| `activity` | `soma`, `dend`, `bg`: components-by-time with matching component counts |
| `optics` | `run_fixed` / `run_optics` output, or `psf` plus applicable masks and tails |
| `scan` | `scan_buff`, `scan_avg`, `sfrac`, `nuc_label`, etc. |
| `motion` | Optional per-frame one-based `x`, `y`, `z`; reuse the recorded path for replay |

After decoding the source structures:

```python
from maomi import attach_rana, RanaAddon
from maomi.movie import scan_volume

# Set pitch from the source dataset's physical voxel size.
local_optics = attach_rana(optics, RanaAddon(voxel_size_um=pitch))
movie = scan_volume(neurons, local_optics, activity, scan, add_noise=False)
```

Native attachment needs `points_um` and `psf_3d` from propagation. A MAT file
containing only a mean PSF cannot supply missing local shapes: propagate the
source tissue or provide a complete custom field. See [RANA integration](RANA_MODE.md).
Some original volumes were available by request rather than on OSF, as the
paper explains.

For an immediately runnable example with generated cells, activity, movies
and ground truth, no download is needed:

```sh
python examples/rana_movie.py --rana --output artifacts/local_movie
```
