"""Download example data files from OSF (https://osf.io/a2vc7/)."""
import os
import urllib.request
import urllib.error

_EXAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))


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


def download_bistable():
    """Download Brownian dynamics data for bistable potential examples."""
    dest_dir = os.path.join(_EXAMPLES_DIR, "datafiles", "brownian_dynamics")
    os.makedirs(dest_dir, exist_ok=True)
    files = {
        "cossio_kl0_Dx1_Dq1.h5": "https://osf.io/download/fcx6g/",
        "cossio_kl1.3_Dx1_Dq1.h5": "https://osf.io/download/60dc7b3531881a01f2637256/",
    }
    for name, url in files.items():
        dest = os.path.join(dest_dir, name)
        if not os.path.isfile(dest):
            print(f"Downloading {name} ...")
            _download(url, dest)
        else:
            print(f"{name} already present, skipping.")


def download_schutte():
    """Download Brownian dynamics data for Schutte potential example."""
    dest_dir = os.path.join(_EXAMPLES_DIR, "schutte_potential", "data")
    os.makedirs(dest_dir, exist_ok=True)
    files = {
        "schutte_num5e+08_dt0.0001_fwrite10.h5": "https://osf.io/download/zt2pe/",
    }
    for name, url in files.items():
        dest = os.path.join(dest_dir, name)
        if not os.path.isfile(dest):
            print(f"Downloading {name} ...")
            _download(url, dest)
        else:
            print(f"{name} already present, skipping.")


if __name__ == "__main__":
    print("=== Bistable potential ===")
    download_bistable()
    print("=== Schutte potential ===")
    download_schutte()
    print("Done.")
