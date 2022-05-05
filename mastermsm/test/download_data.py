import os
from urllib.request import urlretrieve

def download_osf_alaTB():
    downloads = {"alaTB.gro": "https://osf.io/hgbqs/download", \
            "alaTB.xtc": "https://osf.io/ujmhc/download"}
    cpath = os.getcwd()
    if os.path.exists(cpath+"/test/data") is False:
        os.mkdir(cpath+"/test/data")
    for k, v in downloads.items():
        if os.path.isfile(cpath+"/%s"%k) is False:
            urlretrieve(v, "test/data/" + k)

def download_osf_ala5():
    downloads = {"ala5.gro": "https://osf.io/6uznm/download", \
            "ala5.xtc": "https://osf.io/gmxpy/download"}
    cpath = os.getcwd()
    if os.path.exists(cpath+"/test/data") is False:
        os.mkdir(cpath+"/test/data")
    for k, v in downloads.items():
        if os.path.isfile(cpath+"/%s"%k) is False:
            urlretrieve(v, "test/data/" + k)
