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


def _downloads_disabled():
    """True if $MASTERMSM_NO_DOWNLOAD is set to a non-empty value other
    than 0/false/no (e.g. for offline builds such as conda-forge CI)."""
    value = os.environ.get("MASTERMSM_NO_DOWNLOAD", "").strip().lower()
    return value not in ("", "0", "false", "no")


def _fetch(downloads):
    """Download missing files; skip the calling test if that fails.

    With $MASTERMSM_NO_DOWNLOAD set, missing files are not downloaded
    and the calling test is skipped at once; files already present in
    the data directory are still used.
    """
    missing = [k for k in downloads
               if not os.path.isfile(os.path.join(_DATA_DIR, k))]
    if missing and _downloads_disabled():
        raise unittest.SkipTest(
            "test data missing from %s and downloads disabled by "
            "MASTERMSM_NO_DOWNLOAD" % _DATA_DIR)
    try:
        os.makedirs(_DATA_DIR, exist_ok=True)
        for k in missing:
            _download(downloads[k], os.path.join(_DATA_DIR, k))
    except (urllib.error.URLError, OSError) as e:
        raise unittest.SkipTest(
            "test data could not be downloaded to %s (%s)" % (_DATA_DIR, e))


def download_osf_alaTB():
    _fetch({"alaTB.gro": "https://osf.io/hgbqs/download",
            "alaTB.xtc": "https://osf.io/ujmhc/download"})


def download_osf_ala5():
    _fetch({"ala5.gro": "https://osf.io/6uznm/download",
            "ala5.xtc": "https://osf.io/gmxpy/download"})
