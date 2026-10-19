# Source attribution

MaoMi's tissue, activity, scanning and optical routines derive from NAOMi by
Alex Song, Adam Charles and collaborators. The original MIT notice is included
in `LICENSES/NAOMi-MIT.txt`. Development used an identified MATLAB checkout
containing local modifications; it is not represented as an untouched upstream
release. MATLAB and proprietary toolbox code are not included.

Original paper: Song, Gauthier, Pillow, Tank and Charles (2021),
*Neural anatomy and optical microscopy (NAOMi) simulation for evaluating calcium
imaging methods*, Journal of Neuroscience Methods 358, 109173.
[DOI](https://doi.org/10.1016/j.jneumeth.2021.109173).

The optional RayOptics adapter uses the public RayGrid interface of RayOptics
by Michael J. Hayford and contributors: https://github.com/mjhoptics/ray-optics.
RayOptics is installed separately. The example lens is illustrative.

NumPy, SciPy, PyTorch, CuPy, Numba, Matplotlib, tifffile, Astropy and
imageio-ffmpeg are installed separately under their own licenses. No third-party
binary libraries, MATLAB/MEX binaries or FFmpeg executable are bundled.

RANA's field integral and interface equations reference the PSF Generator
theory documentation; the implementation is original to MaoMi and does not
bundle PSF Generator source. See [model references](docs/RANA_PHYSICS.md).
The pixel-cat asset was generated from the project owner's own cat photograph;
the original photograph is not distributed.
