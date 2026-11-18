"""Download external example datasets from Zenodo (via mddatabiokt) and OSF."""

import os
import urllib.error
import urllib.request

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


def download_zenodo():
    """Fetch Zenodo v1 datasets (alaTB, ala5, muller) into the mddatabiokt cache."""
    try:
        import mddatabiokt as mdd
    except ImportError:
        print(
            "mddatabiokt not installed; skipping Zenodo datasets "
            "(install with: pip install git+https://github.com/BioKT/MDdata.git)."
        )
        return

    datasets = [
        ("alaTB", "gromacs_ff99sbildn", ("trajs", "smoke")),
        ("alaTB", "gromacs_ff03star", ("trajs",)),
        ("ala5", "gromacs_eq", ("trajs",)),
        ("muller", "bd", ("trajs",)),
    ]
    for system, variant, subsets in datasets:
        print(f"Fetching {system} ({variant}) via mddatabiokt ...")
        ds = mdd.fetch(system, variant=variant, subsets=subsets)
        print(f"  cached in {ds.cache_dir}")


def download_bistable():
    """Download Brownian dynamics data for bistable potential examples (OSF)."""
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
    """Download Brownian dynamics data for Schutte potential example (OSF)."""
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
    print("=== Zenodo datasets (mddatabiokt) ===")
    download_zenodo()
    print("=== Bistable potential (OSF) ===")
    download_bistable()
    print("=== Schutte potential (OSF) ===")
    download_schutte()
    print("Done.")
