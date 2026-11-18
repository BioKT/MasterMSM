# Testing

The test suite uses `pytest`. Some trajectory tests use reference
simulation datasets (`alaTB`, `ala5`, `muller`) archived on
[Zenodo](https://doi.org/10.5281/zenodo.24253206) and fetched via
[`mddatabiokt`](https://github.com/BioKT/MDdata) into the local cache
(`$MASTERMSM_TEST_DATA`, `$MDDATABIOKT_DIR`, or
`~/.cache/mddatabiokt`) on the first run. If `mddatabiokt` is not
installed or `MASTERMSM_NO_DOWNLOAD=1` is set without cached files,
those tests are skipped automatically.

From the root of the repository:

```bash
pip install pytest pytest-cov git+https://github.com/BioKT/MDdata.git
pytest mastermsm/test/
```

To also report coverage:

```bash
pytest mastermsm/test/ --cov=mastermsm --cov-report=term
```

The same tests run on GitHub Actions for Python 3.10-3.12 on Linux and
macOS (see `.github/workflows/ci.yml`).
