import os
from urllib.request import urlretrieve

def download_test_data():
    base_url = "https://mastermsm.s3.eu-west-2.amazonaws.com/"
    gro = "test/data/alaTB.gro"
    xtc = "test/data/protein_only.xtc"
    cpath = os.getcwd()
    if os.path.exists(cpath+"/test/data") is False:
        os.mkdir(cpath+"/test/data")
    for fname in [gro,xtc]:
        if os.path.isfile(cpath+"/%s"%fname) is False:
            urlretrieve(base_url+fname, fname)
