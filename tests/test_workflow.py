import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
from scipy.io import savemat
from maomi import (Scene,OpticalConfig,load_naomi_mat,save_scene,load_scene,
                   RunConfig,run_scene,MovieDataset,prepare_optics,RanaAddon)
from maomi.archive import save_optics,load_optics
from maomi.mat_io import MatArray,inspect_naomi_mat
from maomi.movie import scan_volume
from maomi._movie_buffers import SparseMovieBuffer
import torch
from test_rana_addon import fixture,addon


def sample_mat():
    shape = (6,8,4)
    ids = np.array([1,2,7,13,50,100,192])
    cells = np.empty((2,3),object)
    for i in range(2):
        cells[i] = (ids+i if i==0 else np.array([3,22,84]),
                    np.linspace(.2,1,len(ids) if i==0 else 3),
                    np.ones(len(ids) if i==0 else 3,bool))
    nucleus = np.empty((2,2),object)
    nucleus[0] = (np.array([2,7]),np.ones(2))
    nucleus[1] = (np.array([3]),np.ones(1))
    return dict(vol_out=dict(neur_vol=np.arange(np.prod(shape),dtype=np.float32).reshape(shape,order="F"),
                             gp_vals=cells,gp_nuc=nucleus,bg_proc=np.empty((0,2),object)),
                vol_params=dict(vres=2.,vol_depth=10.,vol_sz=np.array(shape)/2),
                psf_params=dict(psf_sz=[2,2,2],taillength=2,scaling="two-photon",
                                zernikeWt=[0,0,0,0,.1,0,0,0,0,0,.12]),
                activity=dict(soma=np.arange(10,dtype=float).reshape(2,5)+1,dend=np.ones((2,5))))


def write_hdf(path,data):
    import h5py
    with h5py.File(path,"w") as f:
        refs = f.create_group("#refs#")
        def put(group,key,value):
            if isinstance(value,dict):
                g = group.create_group(key)
                g.attrs["MATLAB_class"] = b"struct"
                for k,v in value.items():
                    put(g,k,v)
            elif isinstance(value,str):
                d = group.create_dataset(key,data=np.array([ord(c) for c in value],np.uint16)[:,None])
                d.attrs["MATLAB_class"] = b"char"
            else:
                a = np.asarray(value)
                if a.dtype==object:
                    d = group.create_dataset(key,shape=a.T.shape,dtype=h5py.ref_dtype)
                    d.attrs["MATLAB_class"] = b"cell"
                    for idx in np.ndindex(a.shape):
                        name = str(len(refs))
                        put(refs,name,np.atleast_2d(a[idx]))
                        d[tuple(reversed(idx))] = refs[name].ref
                else:
                    d = group.create_dataset(key,data=np.atleast_2d(a).T)
                    d.attrs["MATLAB_class"] = b"logical" if a.dtype==bool else b"double"
        for k,v in data.items():
            put(f,k,v)


@unittest.skipUnless(importlib.util.find_spec("h5py"),"Install the mat extra")
class WorkflowTests(unittest.TestCase):
    def test_v5_v73_crop_and_sparse_index_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p,q = Path(tmp)/"v5.mat",Path(tmp)/"v73.mat"
            source = sample_mat()
            savemat(p,source)
            write_hdf(q,source)
            roi = ((1,5),(0,7),(0,3))
            a,b = [load_naomi_mat(x,roi=roi) for x in (p,q)]
            self.assertIsInstance(b.neurons["neur_vol"],MatArray)
            np.testing.assert_array_equal(a.neurons["neur_vol"],np.asarray(b.neurons["neur_vol"]))
            for ar,br in zip(a.neurons["gp_vals"],b.neurons["gp_vals"]):
                for x,y in zip(ar,br):
                    np.testing.assert_array_equal(x,y)
            # Original ID 2 is (x=1,y=0,z=0), hence cropped ID 1.
            self.assertEqual(a.neurons["gp_vals"][0][0][0],1)
            self.assertEqual(len(b.neurons["gp_vals"]),2)
            np.testing.assert_array_equal(a.activity["soma"],b.activity["soma"])
            self.assertEqual(b.config.volume_um,(2.,3.5,1.5))
            self.assertEqual(b.config.scaling,"two-photon")
            self.assertEqual(b.config.zernike_weights[4],.1)
            self.assertEqual(a.config.zernike_weights,b.config.zernike_weights)
            self.assertIn("vol_out",inspect_naomi_mat(q))
            save_scene(b,Path(tmp)/"scene.h5")
            c = load_scene(Path(tmp)/"scene.h5")
            np.testing.assert_array_equal(c.neurons["neur_vol"],a.neurons["neur_vol"])
            np.testing.assert_array_equal(c.neurons["gp_vals"][1][0],a.neurons["gp_vals"][1][0])

    def test_single_cell_and_bad_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"single.mat"
            d=sample_mat()
            d["vol_out"]["gp_vals"]=d["vol_out"]["gp_vals"][:1]
            savemat(p,d)
            self.assertEqual(len(load_naomi_mat(p).neurons["gp_vals"]),1)
            with self.assertRaises(ValueError):
                load_naomi_mat(p,roi=((-1,2),(0,3),(0,2)))
            savemat(p,dict(analysis=np.ones(3)))
            with self.assertRaises(ValueError):
                load_naomi_mat(p)

    def test_v73_empty_cell_encoding(self):
        import h5py
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"empty.mat"
            write_hdf(p,sample_mat())
            with h5py.File(p,"a") as f:
                del f["vol_out/bg_proc"]
                ds=f["vol_out"].create_dataset("bg_proc",data=np.array([0,0],np.uint64))
                ds.attrs["MATLAB_empty"]=1
                ds.attrs["MATLAB_class"]=b"cell"
            s=load_naomi_mat(p)
            self.assertEqual(list(s.neurons["bg_proc"]),[])

    def test_resume_matches_uninterrupted_with_rana_noise_and_motion(self):
        optics,neurons,activity,motion=fixture()
        optics=addon(optics)
        cfg=OpticalConfig(volume_um=(12,14,9),psf_um=(3,4,3))
        scene=Scene(neurons,cfg,activity,scan=dict(scan_buff=2,scan_avg=1,sfrac=2))
        settings=RunConfig(device="cpu",seed=42,chunk_frames=2)
        expected=run_scene(scene,settings=settings,optics=optics,motion=motion)
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"movie"
            partial=run_scene(scene,settings=settings,optics=optics,motion=motion,output=out,max_frames=1)
            self.assertEqual(len(partial),1)
            self.assertFalse(partial.complete)
            result=run_scene(output=out,resume=True)
            self.assertTrue(result.complete)
            checked=MovieDataset(out,verify=True)
            for i in range(len(checked)):
                np.testing.assert_array_equal(checked[i]["clean"],expected["clean"][...,i].T)
                np.testing.assert_array_equal(checked[i]["noisy"],expected["noisy"][...,i].T)
            restored=checked.scene()
            np.testing.assert_array_equal(restored.activity["soma"],activity["soma"])
            self.assertIn("motion",checked.metadata)

    def test_sparse_overwrite_add_and_streamed_activity(self):
        ids=np.array([1,1,3],np.int64)
        groups=([(ids,np.array([2,3,4],np.float32)),(np.array([1]),np.array([7],np.float32))],False)
        bg=([(np.array([1,1]),np.array([.5,.25],np.float32))],True)
        for backend in ("numpy","auto"):
            writer=SparseMovieBuffer((2,2,2),(groups,bg),backend=backend)
            out=writer.frame([np.array([1,2,.5],np.float32)]).ravel(order="F").copy()
            self.assertEqual(out[1],14.375)
            np.testing.assert_array_equal(writer.frame([np.zeros(3)]),np.zeros((2,2,2)))
        class Bounded:
            shape=(2,3)
            def __array__(self,*args,**kwargs):
                raise AssertionError("Activity must be read in chunks")
            def __getitem__(self,k):
                return np.ones(self.shape)[k]
        optics,neurons,activity,motion=fixture()
        activity["soma"]=Bounded()
        scan_volume(neurons,optics,activity,dict(scan_buff=2,scan_avg=1,sfrac=1,chunk_frames=1),
                    motion=motion,add_noise=False)

    def test_dev0_archive_resume_and_unknown_version_rejection(self):
        import json
        optics, neurons, activity, motion = fixture()
        scene = Scene(neurons, OpticalConfig(volume_um=(12,14,9), psf_um=(3,4,3)),
                      activity, scan=dict(scan_buff=2, scan_avg=1, sfrac=2))
        settings = RunConfig(device="cpu", seed=42, chunk_frames=1)
        expected = run_scene(scene, settings=settings, optics=optics, motion=motion)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"movie"
            run_scene(scene, settings=settings, optics=optics, motion=motion, output=path, max_frames=1)
            metadata = path/"run.json"
            record = json.loads(metadata.read_text(encoding="utf8"))
            record["maomi_version"] = "99.0.0"
            metadata.write_text(json.dumps(record), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "version"):
                run_scene(output=path, resume=True)
            record["maomi_version"] = "0.8.0.dev0"
            metadata.write_text(json.dumps(record), encoding="utf8")
            resumed = run_scene(output=path, resume=True)
            self.assertTrue(resumed.complete)
            for i in range(3):
                for key in ("clean", "noisy"):
                    np.testing.assert_array_equal(resumed[i][key], expected[key][...,i].T)

    def test_cpu_prepare_no_static_image_and_reusable_archive(self):
        cfg=OpticalConfig(volume_um=(8,8,6),depth_um=5,tail_um=2,psf_um=(4,4,4))
        out=prepare_optics(cfg,device="cpu",count_per_axis=2,rana=RanaAddon(status_window=False))
        self.assertNotIn("image",out)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"optics.h5"
            save_optics(out,p)
            loaded=load_optics(p)
            volume=np.ones((8,8,6))
            np.testing.assert_array_equal(loaded["rana"].static(volume),out["rana"].static(volume))

    def test_lazy_vessels_and_cpu_static_multibeam(self):
        from maomi.optics import prepare_fields,collection_mask
        from maomi import simulate
        cfg=OpticalConfig(volume_um=(8,8,6),depth_um=5,tail_um=2,psf_um=(4,4,4))
        shape=np.rint(np.array(cfg.parameters()[0].vasc_sz)*cfg.vres).astype(int)
        shape[2]=max(shape[2],12)
        data=np.random.default_rng(11).random(shape)<.03
        class Bounded:
            shape=data.shape
            def __array__(self,*args,**kwargs):
                raise AssertionError("Vessels must be read as slabs")
            def __getitem__(self,k):
                return data[k]
        for a,b in zip(prepare_fields(cfg,data,scatter=False),prepare_fields(cfg,Bounded(),scatter=False)):
            np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(collection_mask(data,cfg),collection_mask(Bounded(),cfg))
        from maomi.optics import pupil
        u=pupil(cfg)
        result=simulate(np.ones((8,8,6)),config=cfg,device="cpu",count_per_axis=2,
                        input_pupil={"a":u,"b":u},beam_weights={"a":.25,"b":.75})
        self.assertEqual(result["image"].shape,(8,8,6))

    def test_imported_zernike_and_explicit_phase_override(self):
        from maomi.optics import pupil
        from dataclasses import replace
        cfg=OpticalConfig(volume_um=(8,8,6),depth_um=5,tail_um=2,psf_um=(4,4,4))
        aberrated=replace(cfg,zernike_weights=(0,0,0,0,.1,0,0,0,0,0,.12))
        a,b=pupil(cfg).numpy(),pupil(aberrated).numpy()
        self.assertGreater(np.linalg.norm(a-b)/np.linalg.norm(a),.01)
        # Original MATLAB gives explicit custom_phase precedence.
        np.testing.assert_array_equal(pupil(aberrated,custom_phase=np.zeros((9,9))).numpy(),a)
    @unittest.skipUnless(torch.cuda.is_available() and importlib.util.find_spec("cupy"),"CUDA optional")
    def test_cuda_prepare_does_not_render_discarded_image(self):
        cfg=OpticalConfig(volume_um=(8,8,6),depth_um=5,tail_um=2,psf_um=(4,4,4))
        with patch("maomi.fixed.zstack",side_effect=AssertionError("Unused static image")):
            out=prepare_optics(cfg,device="cuda",count_per_axis=2,batch_size=2,
                               rana=RanaAddon(status_window=False))
        self.assertNotIn("image",out)
        self.assertIn("rana",out)

    def test_missing_tissue_extra_has_install_instructions(self):
        from maomi import generate_scene
        with patch.dict("sys.modules", {"numba": None}):
            with self.assertRaisesRegex(ImportError, r"\.\[tissue\]"):
                generate_scene(seed=19)

    @unittest.skipUnless(importlib.util.find_spec("numba"), "Install the tissue extra")
    def test_compact_tissue_has_complete_parameters_and_finite_outputs(self):
        from maomi import generate_scene
        scene=generate_scene(seed=19)
        self.assertTrue(np.isfinite(scene.neurons["neur_vol"]).all())
        self.assertGreater(np.count_nonzero(scene.neurons["neur_vol"]),0)
        self.assertGreater(len(scene.neurons["gp_vals"]),0)
        scene.generate_activity(5,seed=3)
        self.assertEqual(scene.activity["soma"].shape[1],5)

if __name__=="__main__":
    unittest.main()
