import os
import tempfile
import unittest
from unittest import mock

from mastermsm.test import download_data


class TestNoDownload(unittest.TestCase):
    """MASTERMSM_NO_DOWNLOAD skips data tests without network access."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = mock.patch.object(download_data, "_DATA_DIR",
                                    self.tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.files = {"a.gro": "https://example.invalid/a",
                      "a.xtc": "https://example.invalid/b"}

    def test_flag_values(self):
        for value, disabled in [("", False), ("0", False),
                                ("false", False), ("No", False),
                                ("1", True), ("yes", True),
                                ("TRUE", True)]:
            with mock.patch.dict(os.environ,
                                 {"MASTERMSM_NO_DOWNLOAD": value}):
                self.assertEqual(download_data._downloads_disabled(),
                                 disabled, value)

    def test_skips_without_download(self):
        with mock.patch.dict(os.environ, {"MASTERMSM_NO_DOWNLOAD": "1"}), \
                mock.patch.object(download_data, "_download") as dl:
            with self.assertRaises(unittest.SkipTest):
                download_data._fetch(self.files)
            dl.assert_not_called()

    def test_uses_local_files(self):
        for k in self.files:
            open(os.path.join(self.tmp.name, k), "w").close()
        with mock.patch.dict(os.environ, {"MASTERMSM_NO_DOWNLOAD": "1"}), \
                mock.patch.object(download_data, "_download") as dl:
            download_data._fetch(self.files)  # no SkipTest
            dl.assert_not_called()

    def test_downloads_only_missing(self):
        open(os.path.join(self.tmp.name, "a.gro"), "w").close()
        env = {k: v for k, v in os.environ.items()
               if k != "MASTERMSM_NO_DOWNLOAD"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(download_data, "_download") as dl:
            download_data._fetch(self.files)
            dl.assert_called_once_with(
                "https://example.invalid/b",
                os.path.join(self.tmp.name, "a.xtc"))


if __name__ == "__main__":
    unittest.main()
