# Testing

The test suite uses `pytest`. Some tests need short MD trajectories of
the alanine dipeptide and pentapeptide, which are downloaded from
[OSF](https://osf.io/a2vc7) into `mastermsm/test/data/` on the first
run.

From the root of the repository:

```bash
pip install pytest pytest-cov
pytest mastermsm/test/
```

To also report coverage:

```bash
pytest mastermsm/test/ --cov=mastermsm --cov-report=term
```

The same tests run on GitHub Actions for Python 3.10-3.12 on Linux and
macOS (see `.github/workflows/ci.yml`).
