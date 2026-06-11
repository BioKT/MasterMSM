import os
import urllib.request
import urllib.error

# Data directory lives next to this file, regardless of where tests are run from
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _download(url, dest, max_redirects=10):
    """Download url to dest, following HTTP 308 (Python 3.10 urllib omits it)."""
    for _ in range(max_redirects):
        try:
            urllib.request.urlretrieve(url, dest)
            return
        except urllib.error.HTTPError as e:
            if e.code == 308 and e.headers.get("Location"):
                url = e.headers["Location"]
            else:
                raise


def download_osf_alaTB():
    downloads = {"alaTB.gro": "https://osf.io/hgbqs/download",
                 "alaTB.xtc": "https://osf.io/ujmhc/download"}
    os.makedirs(_DATA_DIR, exist_ok=True)
    for k, v in downloads.items():
        dest = os.path.join(_DATA_DIR, k)
        if not os.path.isfile(dest):
            _download(v, dest)

def download_osf_ala5():
    downloads = {"ala5.gro": "https://osf.io/6uznm/download",
                 "ala5.xtc": "https://osf.io/gmxpy/download"}
    os.makedirs(_DATA_DIR, exist_ok=True)
    for k, v in downloads.items():
        dest = os.path.join(_DATA_DIR, k)
        if not os.path.isfile(dest):
            _download(v, dest)
