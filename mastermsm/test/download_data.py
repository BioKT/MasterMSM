"""Fetch verified benchmark simulation datasets via ``mddatabiokt`` (Zenodo).

Environment variables:

* ``MASTERMSM_TEST_DATA``: override the ``mddatabiokt`` cache directory.
  When unset, ``mddatabiokt`` resolves ``$MDDATABIOKT_DIR``, ``$MDDATA_DIR``,
  ``$XDG_CACHE_HOME/mddatabiokt``, or ``~/.cache/mddatabiokt``.
* ``MASTERMSM_NO_DOWNLOAD``: when set to a non-empty value other than
  ``0``/``false``/``no``, network downloads are disabled (``offline=True``)
  and tests whose cached files are missing are skipped immediately.

If ``mddatabiokt`` is not installed, tests requiring external simulation
datasets are skipped gracefully so offline or minimal environments remain
testable.
"""

import os
import unittest
import urllib.error


def _cache_dir():
    """Custom cache directory from ``$MASTERMSM_TEST_DATA``, or ``None``."""
    env = os.environ.get("MASTERMSM_TEST_DATA", "").strip()
    return env or None


def _downloads_disabled():
    """True if ``$MASTERMSM_NO_DOWNLOAD`` is set to a non-empty value other
    than 0/false/no (e.g. for offline builds such as conda-forge CI).
    """
    value = os.environ.get("MASTERMSM_NO_DOWNLOAD", "").strip().lower()
    return value not in ("", "0", "false", "no")


def _require_mddata():
    """Import and return ``mddatabiokt``, or skip the calling test."""
    try:
        import mddatabiokt as mdd
        return mdd
    except ImportError as exc:
        raise unittest.SkipTest(
            "mddatabiokt is not installed; install BioKT/MDdata to run "
            "external-data tests"
        ) from exc


def _call_mddata(func, *args, **kwargs):
    """Run an ``mddatabiokt`` loader with MasterMSM's cache/offline settings."""
    mdd = _require_mddata()
    cache_dir = _cache_dir()
    offline = _downloads_disabled()
    try:
        return func(*args, cache_dir=cache_dir, offline=offline, **kwargs)
    except FileNotFoundError as exc:
        if offline:
            raise unittest.SkipTest(
                "test data missing from cache and downloads disabled by "
                "MASTERMSM_NO_DOWNLOAD (%s)" % exc
            ) from exc
        raise unittest.SkipTest("test data file missing (%s)" % exc) from exc
    except (urllib.error.URLError, OSError, mdd.MDDataError) as exc:
        raise unittest.SkipTest(
            "test data could not be fetched via mddatabiokt (%s)" % exc
        ) from exc


def fetch_alaTB(variant="gromacs_ff99sbildn", subset="smoke"):
    """Return ``(top_path, xtc_path)`` for the Zenodo ``alaTB`` dataset.

    Parameters
    ----------
    variant : str
        ``"gromacs_ff99sbildn"`` (default) or ``"gromacs_ff03star"``.
    subset : str
        ``"smoke"`` (10 ns ``protein_only.xtc``, available for
        ``gromacs_ff99sbildn``) or ``"trajs"`` (full production trajectory).
    """
    mdd = _require_mddata()
    if subset == "smoke":
        bundle = _call_mddata(
            mdd.fetch, "alaTB", variant=variant, subsets=("trajs", "smoke")
        )
        return str(bundle.topology), str(bundle.files["protein_only.xtc"])
    if subset == "trajs":
        bundle = _call_mddata(
            mdd.fetch, "alaTB", variant=variant, subsets=("trajs",)
        )
        return str(bundle.topology), str(bundle.trajectories[0])
    raise ValueError("unknown alaTB subset: %r" % (subset,))


def fetch_ala5(variant="gromacs_eq"):
    """Return ``(top_path, xtc_path)`` for the Zenodo ``ala5`` dataset."""
    mdd = _require_mddata()
    bundle = _call_mddata(
        mdd.fetch, "ala5", variant=variant, subsets=("trajs",)
    )
    return str(bundle.topology), str(bundle.trajectories[0])


def fetch_muller(variant="bd"):
    """Return the array dictionary for the Zenodo 2D Mueller-Brown BD dataset."""
    mdd = _require_mddata()
    data = _call_mddata(
        mdd.load_numpy, "muller", variant=variant, subset="trajs"
    )
    return data["muller_bd_trajs"]
