import os
import pathlib
import sys
import types
import unittest
import urllib.error
from unittest import mock

from mastermsm.test import download_data


class _DummyMDDataError(Exception):
    pass


def _fake_mdd_module(fetch_fn=None, load_numpy_fn=None):
    mod = types.ModuleType("mddatabiokt")
    mod.MDDataError = _DummyMDDataError
    mod.fetch = fetch_fn or mock.MagicMock()
    mod.load_numpy = load_numpy_fn or mock.MagicMock()
    return mod


class TestDownloadData(unittest.TestCase):
    """Tests for mddatabiokt integration, offline opt-out, and graceful skips."""

    def test_flag_values(self):
        for value, disabled in [
            ("", False),
            ("0", False),
            ("false", False),
            ("No", False),
            ("1", True),
            ("yes", True),
            ("TRUE", True),
        ]:
            with mock.patch.dict(os.environ, {"MASTERMSM_NO_DOWNLOAD": value}):
                self.assertEqual(
                    download_data._downloads_disabled(), disabled, value
                )

    def test_cache_dir_env(self):
        env = {k: v for k, v in os.environ.items() if k != "MASTERMSM_TEST_DATA"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsNone(download_data._cache_dir())
        with mock.patch.dict(os.environ, {"MASTERMSM_TEST_DATA": "/tmp/mdd_test"}):
            self.assertEqual(download_data._cache_dir(), "/tmp/mdd_test")

    def test_skips_when_mddatabiokt_missing(self):
        with mock.patch.dict(sys.modules, {"mddatabiokt": None}):
            with self.assertRaises(unittest.SkipTest) as ctx:
                download_data.fetch_alaTB()
            self.assertIn("mddatabiokt is not installed", str(ctx.exception))

    def test_skips_without_download_when_offline_and_missing(self):
        fetch_mock = mock.MagicMock(
            side_effect=FileNotFoundError("Offline mode enabled")
        )
        fake_mdd = _fake_mdd_module(fetch_fn=fetch_mock)
        with mock.patch.dict(sys.modules, {"mddatabiokt": fake_mdd}), \
                mock.patch.dict(
                    os.environ,
                    {
                        "MASTERMSM_NO_DOWNLOAD": "1",
                        "MASTERMSM_TEST_DATA": "/tmp/custom_cache",
                    },
                ):
            with self.assertRaises(unittest.SkipTest) as ctx:
                download_data.fetch_alaTB()
            self.assertIn("MASTERMSM_NO_DOWNLOAD", str(ctx.exception))
            fetch_mock.assert_called_once_with(
                "alaTB",
                variant="gromacs_ff99sbildn",
                subsets=("trajs", "smoke"),
                cache_dir="/tmp/custom_cache",
                offline=True,
            )

    def test_skips_on_network_or_mddata_error(self):
        for err in (
            urllib.error.URLError("network unreachable"),
            _DummyMDDataError("checksum mismatch"),
        ):
            fetch_mock = mock.MagicMock(side_effect=err)
            fake_mdd = _fake_mdd_module(fetch_fn=fetch_mock)
            env = {
                k: v
                for k, v in os.environ.items()
                if k not in ("MASTERMSM_NO_DOWNLOAD", "MASTERMSM_TEST_DATA")
            }
            with mock.patch.dict(sys.modules, {"mddatabiokt": fake_mdd}), \
                    mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(unittest.SkipTest):
                    download_data.fetch_ala5()

    def test_fetch_alaTB_subsets_and_variants(self):
        bundle = types.SimpleNamespace(
            topology=pathlib.Path("/cache/alaTB.gro"),
            trajectories=[pathlib.Path("/cache/alatb_n1_ppn24.xtc")],
            files={"protein_only.xtc": pathlib.Path("/cache/protein_only.xtc")},
        )
        fetch_mock = mock.MagicMock(return_value=bundle)
        fake_mdd = _fake_mdd_module(fetch_fn=fetch_mock)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("MASTERMSM_NO_DOWNLOAD", "MASTERMSM_TEST_DATA")
        }
        with mock.patch.dict(sys.modules, {"mddatabiokt": fake_mdd}), \
                mock.patch.dict(os.environ, env, clear=True):
            top, xtc = download_data.fetch_alaTB()
            self.assertEqual((top, xtc), ("/cache/alaTB.gro", "/cache/protein_only.xtc"))

            top2, xtc2 = download_data.fetch_alaTB(
                variant="gromacs_ff03star", subset="trajs"
            )
            self.assertEqual(
                (top2, xtc2), ("/cache/alaTB.gro", "/cache/alatb_n1_ppn24.xtc")
            )

            with self.assertRaises(ValueError):
                download_data.fetch_alaTB(subset="invalid")

    def test_fetch_ala5_and_muller(self):
        ala5_bundle = types.SimpleNamespace(
            topology=pathlib.Path("/cache/ala5.gro"),
            trajectories=[pathlib.Path("/cache/ala5.xtc")],
        )
        muller_arrays = {"coords": [[0.0, 1.0]]}
        fake_mdd = _fake_mdd_module(
            fetch_fn=mock.MagicMock(return_value=ala5_bundle),
            load_numpy_fn=mock.MagicMock(
                return_value={"muller_bd_trajs": muller_arrays}
            ),
        )
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("MASTERMSM_NO_DOWNLOAD", "MASTERMSM_TEST_DATA")
        }
        with mock.patch.dict(sys.modules, {"mddatabiokt": fake_mdd}), \
                mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                download_data.fetch_ala5(), ("/cache/ala5.gro", "/cache/ala5.xtc")
            )
            self.assertIs(download_data.fetch_muller(), muller_arrays)


if __name__ == "__main__":
    unittest.main()
