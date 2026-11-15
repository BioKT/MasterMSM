import os
import unittest
import urllib.request
import urllib.error


def _data_dir():
    """Directory for the downloaded test trajectories.

    In order of preference: $MASTERMSM_TEST_DATA, the ``data`` folder
    next to this file (source checkouts and writable installs), or a
    per-user cache (read-only installs, e.g. system or conda packages).
    """
    env = os.environ.get("MASTERMSM_TEST_DATA")
    if env:
        return env
    here = os.path.join(os.path.dirname(__file__), "data")
    if os.path.isdir(here) or os.access(os.path.dirname(here), os.W_OK):
        return here
    cache = os.environ.get("XDG_CACHE_HOME",
                           os.path.join(os.path.expanduser("~"), ".cache"))
    return os.path.join(cache, "mastermsm", "test-data")


_DATA_DIR = _data_dir()


def _download(url, dest, max_redirects=10):
    """Download url to dest, following HTTP 308 (Python 3.10 urllib omits it).

    Writes to a temporary file first so that an interrupted download
    never leaves a truncated file at dest.
    """
    part = dest + ".part"
    try:
        for _ in range(max_redirects):
            try:
                urllib.request.urlretrieve(url, part)
                os.replace(part, dest)
                return
            except urllib.error.HTTPError as e:
                if e.code == 308 and e.headers.get("Location"):
                    url = e.headers["Location"]
                else:
                    raise
        raise urllib.error.URLError("too many redirects: %s" % url)
    finally:
        if os.path.exists(part):
            os.remove(part)


def _fetch(downloads):
    """Download missing files; skip the calling test if that fails."""
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        for k, v in downloads.items():
            dest = os.path.join(_DATA_DIR, k)
            if not os.path.isfile(dest):
                _download(v, dest)
    except (urllib.error.URLError, OSError) as e:
        raise unittest.SkipTest(
            "test data could not be downloaded to %s (%s)" % (_DATA_DIR, e))


def download_osf_alaTB():
    _fetch({"alaTB.gro": "https://osf.io/hgbqs/download",
            "alaTB.xtc": "https://osf.io/ujmhc/download"})


def download_osf_ala5():
    _fetch({"ala5.gro": "https://osf.io/6uznm/download",
            "ala5.xtc": "https://osf.io/gmxpy/download"})
