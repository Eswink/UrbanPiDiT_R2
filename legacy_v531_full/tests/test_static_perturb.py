import os
import tempfile
import unittest

import numpy as np
import torch

from data.loader import BeijingWeatherDataset, compute_train_normalization_stats


def _make_ds(*, mode: str, seed: int = 42, deterministic: bool = True, static_vars=None, static_mean=None, static_std=None, k=None):
    ds = BeijingWeatherDataset.__new__(BeijingWeatherDataset)
    ds.include_static = True
    ds.static_vars = list(static_vars or ["a", "b"])
    ds.static_mean = static_mean
    ds.static_std = static_std
    ds.static_perturb = {"mode": mode, "seed": int(seed), "deterministic": bool(deterministic)}
    ds.static_perturb_categorical_modes = {}
    if k is not None:
        ds.static_perturb["k"] = int(k)
    ds.static_perturb_mode = str(ds.static_perturb.get("mode", "none")).lower()
    ds.static_perturb_seed = int(ds.static_perturb.get("seed", 42))
    ds.static_perturb_deterministic = bool(ds.static_perturb.get("deterministic", True))
    ds.continuous_static_vars = [name for name in ds.static_vars if name != "landcover"]
    ds.categorical_static_vars = [name for name in ds.static_vars if name == "landcover"]
    ds._static_perturb_warned = set()
    return ds


class TestStaticPerturb(unittest.TestCase):
    def test_shuffle_hw_is_deterministic_and_value_preserving(self):
        ds = _make_ds(mode="shuffle_hw", seed=123, deterministic=True)
        S = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)

        out1 = ds._apply_static_perturb(S.copy(), idx=7)
        out2 = ds._apply_static_perturb(S.copy(), idx=7)
        self.assertTrue(np.allclose(out1, out2))

        rng = np.random.RandomState(123 + 7)
        perm = rng.permutation(3 * 4)
        flat = S.transpose(1, 2, 0).reshape(12, 2)
        expected = flat[perm].reshape(3, 4, 2).transpose(2, 0, 1)
        self.assertTrue(np.allclose(out1, expected))

        self.assertTrue(np.allclose(np.sort(out1.reshape(-1)), np.sort(S.reshape(-1))))

    def test_fill_mean_returns_zeros(self):
        ds = _make_ds(mode="fill_mean")
        S = np.random.randn(3, 5, 6).astype(np.float32)
        out = ds._apply_static_perturb(S, idx=0)
        self.assertTrue(np.allclose(out, 0.0))

    def test_fill_zero_uses_normed_zero(self):
        static_vars = ["x", "y"]
        static_mean = {"x": np.array([2.0], dtype=np.float32), "y": np.array([-1.0], dtype=np.float32)}
        static_std = {"x": np.array([4.0], dtype=np.float32), "y": np.array([2.0], dtype=np.float32)}
        ds = _make_ds(mode="fill_zero", static_vars=static_vars, static_mean=static_mean, static_std=static_std)
        S = np.ones((2, 3, 3), dtype=np.float32)
        out = ds._apply_static_perturb(S, idx=0)
        self.assertTrue(np.allclose(out[0], -2.0 / 4.0))
        self.assertTrue(np.allclose(out[1], 1.0 / 2.0))

    def test_rot90(self):
        ds = _make_ds(mode="rot90", k=1)
        S0 = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        out = ds._apply_static_perturb(S0, idx=0)
        expected = np.array([[[2, 4], [1, 3]]], dtype=np.float32)
        self.assertTrue(np.allclose(out, expected))

    def test_flip_ud(self):
        ds = _make_ds(mode="flip_ud")
        S0 = np.array([[[1, 2], [3, 4]]], dtype=np.float32)
        out = ds._apply_static_perturb(S0, idx=0)
        expected = np.array([[[3, 4], [1, 2]]], dtype=np.float32)
        self.assertTrue(np.allclose(out, expected))

    def test_static_schema_splits_categorical_without_changing_x_ctx(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = tmp
            train_dir = f"{root}/train"

            os.makedirs(train_dir, exist_ok=True)
            for i in range(6):
                arr = np.zeros((12, 2, 2), dtype=np.float32)
                arr[0:7] = float(i + 1)
                arr[7] = np.array([[0, 1], [2, 3]], dtype=np.float32)
                arr[8] = 10.0 + i
                arr[9] = 20.0 + i
                arr[10] = 30.0 + i
                arr[11] = 40.0 + i
                np.save(f"{train_dir}/2020_01_01_{i:02d}.npy", arr)

            static_schema = {
                "categorical": ["landcover"],
                "continuous": ["building_surface", "buildings", "building_volume", "population"],
                "categorical_cardinality": {"landcover": 4},
            }
            dyn_mean, dyn_std, stat_mean, stat_std = compute_train_normalization_stats(
                root,
                ["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"],
                ["landcover", "building_surface", "buildings", "building_volume", "population"],
                include_static=True,
                static_schema=static_schema,
            )
            np.savez(f"{root}/normalize_mean_train.npz", **{k: np.array([v], dtype=np.float32) for k, v in dyn_mean.items()})
            np.savez(f"{root}/normalize_std_train.npz", **{k: np.array([max(v, 1e-6)], dtype=np.float32) for k, v in dyn_std.items()})
            np.savez(f"{root}/normalize_static_mean_train.npz", **{k: np.array([v], dtype=np.float32) for k, v in stat_mean.items()})
            np.savez(f"{root}/normalize_static_std_train.npz", **{k: np.array([max(v, 1e-6)], dtype=np.float32) for k, v in stat_std.items()})

            ds = BeijingWeatherDataset(
                train_dir,
                k=2,
                delta_t=1,
                lead_times=None,
                dynamic_vars=["d2m", "sp", "t2m", "tcc", "tp", "u10", "v10"],
                static_vars=["landcover", "building_surface", "buildings", "building_volume", "population"],
                include_static=True,
                broadcast_static=True,
                normalize_root=root,
                static_schema=static_schema,
            )
            sample = ds[0]

            self.assertEqual(tuple(sample["x_ctx"].shape), (19, 2, 2))
            self.assertEqual(tuple(sample["static_raw"].shape), (5, 2, 2))
            self.assertEqual(tuple(sample["static_cont"].shape), (4, 2, 2))
            self.assertEqual(tuple(sample["static_cat"].shape), (1, 2, 2))
            self.assertEqual(sample["static_cat"].dtype, torch.int64)
            self.assertNotIn("landcover", stat_mean)
            np.testing.assert_array_equal(sample["static_cat"].numpy()[0], np.array([[0, 1], [2, 3]], dtype=np.int64))
    def test_v51_static_perturb_modes_respect_schema(self):
        static_vars = ["landcover", "building_surface", "population"]
        S = np.stack(
            [
                np.array([[1, 2], [3, 4]], dtype=np.float32),
                np.full((2, 2), 10.0, dtype=np.float32),
                np.full((2, 2), 20.0, dtype=np.float32),
            ],
            axis=0,
        )

        landcover_only = _make_ds(mode="landcover_only", static_vars=static_vars)
        out_land = landcover_only._apply_static_perturb(S.copy(), idx=0)
        np.testing.assert_array_equal(out_land[0], S[0])
        self.assertTrue(np.allclose(out_land[1:], 0.0))

        continuous_only = _make_ds(mode="continuous_only", static_vars=static_vars)
        out_cont = continuous_only._apply_static_perturb(S.copy(), idx=0)
        self.assertTrue(np.allclose(out_cont[0], 0.0))
        np.testing.assert_array_equal(out_cont[1:], S[1:])

        remove_population = _make_ds(mode="remove_population", static_vars=static_vars)
        out_remove = remove_population._apply_static_perturb(S.copy(), idx=0)
        np.testing.assert_array_equal(out_remove[0], S[0])
        np.testing.assert_array_equal(out_remove[1], S[1])
        self.assertTrue(np.allclose(out_remove[2], 0.0))

    def test_shuffle_batch_uses_explicit_sources_when_available(self):
        source_a = np.full((2, 2, 2), 3.0, dtype=np.float32)
        source_b = np.full((2, 2, 2), 7.0, dtype=np.float32)
        ds = _make_ds(mode="shuffle_batch", static_vars=["a", "b"])
        ds.static_perturb["shuffle_sources"] = np.stack([source_a, source_b], axis=0)
        S = np.zeros((2, 2, 2), dtype=np.float32)

        out = ds._apply_static_perturb(S, idx=0)

        self.assertIn(float(out.mean()), {3.0, 7.0})

    def test_wrong_city_static_without_source_is_noop(self):
        ds = _make_ds(mode="wrong_city_static", static_vars=["a", "b"])
        S = np.arange(8, dtype=np.float32).reshape(2, 2, 2)

        out = ds._apply_static_perturb(S.copy(), idx=0)

        np.testing.assert_array_equal(out, S)


if __name__ == "__main__":
    unittest.main()

