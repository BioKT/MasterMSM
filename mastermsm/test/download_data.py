import os
from urllib.request import urlretrieve

# Data directory lives next to this file, regardless of where tests are run from
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def download_osf_alaTB():
    downloads = {"alaTB.gro": "https://osf.io/hgbqs/download",
                 "alaTB.xtc": "https://osf.io/ujmhc/download"}
    os.makedirs(_DATA_DIR, exist_ok=True)
    for k, v in downloads.items():
        dest = os.path.join(_DATA_DIR, k)
        if not os.path.isfile(dest):
            urlretrieve(v, dest)

def download_osf_ala5():
    downloads = {"ala5.gro": "https://osf.io/6uznm/download",
                 "ala5.xtc": "https://osf.io/gmxpy/download"}
    os.makedirs(_DATA_DIR, exist_ok=True)
    for k, v in downloads.items():
        dest = os.path.join(_DATA_DIR, k)
        if not os.path.isfile(dest):
            urlretrieve(v, dest)
