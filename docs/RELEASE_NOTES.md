# MaoMi 0.7.0.dev0 — RANA

- **RANA — Ray Aware Nonuniform Acquisition:** explicit `--mode rana`,
  theoretically higher accuracy through full local PSFs and four physics extensions.
- Depth PSF banks with layered refractive-index / coverglass aberration and
  positive depth interpolation; correct source/scan position selection.
- High-NA vectorial Richards–Wolf optics, Jones polarization, longitudinal
  electric field and s/p interface transmission.
- Finite source-voxel and arbitrary weighted scan-dwell quadrature.
- Coherent spectral pulse integration, dispersion, wavelength-dependent
  diffraction and incoherent emission/collection response.
- The same pixel-cat head, now named Rana, with an independent optional window.
- Full-volume numerical convergence, individual ablations and serial timing:
  [RANA comparison](RANA_COMPARISON.md).
- Standard mode defaults to MATLAB v4 assembly and 65x65 propagation sites.
  Its existing 512x512x60 comparison retains 49.1x optical-stage acceleration
  and 0.00667% image difference from MATLAB.
- Existing native tissue/activity/movie interfaces, basic PSF adapters and
  RayOptics 0.9.5 / OpticalGlass <2 compatibility remain available.

Source ZIP and wheel contain runnable code, tests, examples and public docs.
Local run logs, volume arrays, private backups and the original cat photograph
are excluded. The compact comparison table and figure are public results.
