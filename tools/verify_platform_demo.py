"""Independently verify the stored platform example, without RayOptics or Torch."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.signal import fftconvolve


def verify(folder):
    folder = Path(folder)
    manifest = json.loads((folder/'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != sha:
            raise AssertionError(f'Array hash mismatch: {name}')
    with np.load(folder/'psf.npz', allow_pickle=False) as data:
        psf = data['intensity']
        assert str(data['axes']) == 'XY'
        assert abs(psf.sum()-1.) < 1e-12 and psf.min() >= 0
        np.testing.assert_allclose(data['pixel_size_um'], manifest['pixel_size_um'], rtol=0, atol=0)
    errors = []
    with np.load(folder/'pairs.npz', allow_pickle=False) as d:
        np.testing.assert_allclose(d['cell_components'].sum(axis=1), d['latent_density'], rtol=1e-13, atol=1e-13)
        for name in ['latent_density', 'clean_expected_e', 'noisy_e']:
            assert np.isfinite(d[name]).all(), name
        for row in manifest['pairs']:
            i, scene = row['pair_id'], row['scene_id']
            assert d['pair_scene_id'][i] == scene and d['split'][i] == row['split']
            density = d['latent_density'][scene]
            full = fftconvolve(density, psf, mode='full')
            start = np.asarray(psf.shape)//2
            image = full[start[0]:start[0]+density.shape[0], start[1]:start[1]+density.shape[1]]
            reference = np.maximum(image, 0.)*row['detected_e_per_density_unit']+row['background_e']
            err = np.linalg.norm(reference-d['clean_expected_e'][i])/np.linalg.norm(reference)
            assert err < 1e-12, (i, err)
            errors.append(float(err))
            seed = row['noise_seed']
            rng = np.random.default_rng(np.random.SeedSequence(seed['entropy'], spawn_key=seed['spawn_key']))
            replay = rng.poisson(d['clean_expected_e'][i]).astype(float)+rng.normal(0., row['read_sigma_e'], size=reference.shape)
            np.testing.assert_array_equal(replay, d['noisy_e'][i])
        for scene in range(len(d['latent_density'])):
            assert len(set(d['split'][d['pair_scene_id']==scene])) == 1
    result = dict(passed=True, independent_clean_target_checks=len(errors),
        maximum_relative_l2=max(errors), exact_noise_replays=len(errors),
        scene_disjoint_splits=True, file_checksums=len(manifest['files']),
        scope='SciPy convolution, stored-array invariants and deterministic noise replay; not a real-data or MATLAB equivalence claim')
    (folder/'verification.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=Path(__file__).resolve().parents[1]/'artifacts/rayoptics_demo')
    print(json.dumps(verify(parser.parse_args().folder), indent=2))
