"""
This file is part of the MasterMSM package.

"""
#import h5py
import copy
import sys
import math
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt

def load_mdtraj(top=None, xtc=None, stride=None):
    """ Loads trajectories using mdtraj.

    Parameters
    ----------
    top: str
        The topology file, may be a PDB or GRO file.
    xtc : str
        The trajectory filename.

    Returns
    -------
    mdtrajs : list
        A list of mdtraj Trajectory objects.

    """
    return md.load(xtc, top=top, stride=stride)

def compute_rama(traj, shift=False, sincos=False):
    """ Computes Ramachandran angles 

    Parameters
    ----------
    traj : md.trajectory
        An MDtraj trajectory object
    shift : bool
        Whether we want to shift the torsions or not
    sincos : bool
        Whether we are calculating the sines and cosines 

    Returns
    -------
    phi : array
        An array with phi torsions
    psi : array
        An array with psi torsions

    """
    _, phi = md.compute_phi(traj.mdt)
    _, psi = md.compute_psi(traj.mdt)
    if shift:
        return _shift(phi, psi)
    elif sincos:
        return np.column_stack([np.sin(phi), np.cos(phi)]), \
                np.column_stack([np.sin(psi), np.cos(psi)])
    else:
        return phi, psi

def compute_contacts(traj, scheme=None, log=False):
    """ Computes inter-residue contacts

    Parameters
    ----------
    traj : md.trajectory
        An MDtraj trajectory object
    log : bool
        Whether distances are in log scale

    """
    distances, pairs = md.compute_contacts(traj, scheme=scheme)
    if not log:
        return distances
    else:
        return np.log(distances)

def _filter_states(states):
    """
    Fills not-assigned frames (negative labels) when using dbscan or
    hdbscan. Each unassigned frame inherits the label of the previous
    assigned frame; unassigned frames at the start of the trajectory,
    which have no previous frame, inherit the first assigned label.
    The output always has the same length as the input.

    Raises
    ------
    ValueError
        If no frame is assigned.

    """
    states = list(states)
    first = next((s for s in states if s >= 0), None)
    if first is None:
        raise ValueError("no frame was assigned to any cluster")
    fs = []
    prev = first
    for s in states:
        if s >= 0:
            prev = s
        fs.append(prev)
    return fs

def _shift(phi, psi):
    phi_s, psi_s = copy.deepcopy(phi), copy.deepcopy(psi)
    for i in range(len(phi_s)):
        phi_s[i] = [x + 2*np.pi if x < 0 else x for x in phi_s[i]]
    for i in range(len(psi_s)):
        psi_s[i] = [x + 2*np.pi if (x <-2) else x for x in psi_s[i]]
    return phi_s, psi_s

def _inrange(x, lo, hi):
    return 1 if (x > lo and x < hi) else 0

def _inbounds(bounds, phi, psi):
    for i in range(0, len(bounds), 4):
        if _inrange(phi, bounds[i], bounds[i+1]) and _inrange(psi, bounds[i+2], bounds[i+3]):
            return 1
    return 0

def _state(phi, psi, bounds):
    """ Assign phi, psi (in degrees) to a coarse state.

    Parameters
    ----------
    phi : float
        Phi dihedral angle in degrees.
    psi : float
        Psi dihedral angle in degrees.
    bounds : dict
        Dict mapping state labels to [phi_min, phi_max, psi_min, psi_max, ...] bounds.

    Returns
    -------
    k : str
        Assigned state label, or 'O' if unassigned.

    """
    for k, b in bounds.items():
        if _inbounds(b, phi, psi):
            return k, []
    return 'O', [(phi, psi)]

_DEFAULT_BOUNDS = {'A': [-130., -40., -50., 20.],
                   'E': [-180., -40., 125., 180.],
                   'L': [30., 100., -40., 70.]}

def default_bounds(states=None):
    """ Default Buchete-Hummer bounds for the requested states.

    Parameters
    ----------
    states : list, optional
        State labels, from 'A', 'E' and 'L'. Default is all three.

    Returns
    -------
    bounds : dict
        Maps each label to [phi_min, phi_max, psi_min, psi_max]
        in degrees.

    """
    if states is None:
        states = ['A', 'E', 'L']
    unknown = [s for s in states if s not in _DEFAULT_BOUNDS]
    if unknown:
        raise ValueError("no default bounds for states %s"%unknown)
    return {s: list(_DEFAULT_BOUNDS[s]) for s in states}

def _validate_bounds(b):
    """ Check one state's bounds list, returning it as floats. """
    b = [float(x) for x in b]
    if len(b) == 0 or len(b) % 4 != 0:
        raise ValueError("bounds need groups of four values "\
                "(phi_min, phi_max, psi_min, psi_max), got %i"%len(b))
    for i in range(0, len(b), 4):
        for lo, hi in [(b[i], b[i+1]), (b[i+2], b[i+3])]:
            if not lo < hi:
                raise ValueError("lower bound %g is not below "\
                        "upper bound %g"%(lo, hi))
            if lo < -180. or hi > 180.:
                raise ValueError("bounds must lie within "\
                        "[-180, 180] degrees")
    return b

def _overlaps(bounds):
    """ Pairs of states whose rectangles overlap. """
    rects = [(k, b[i:i+4]) for k, b in bounds.items() \
            for i in range(0, len(b), 4)]
    pairs = []
    for n, (k1, r1) in enumerate(rects):
        for k2, r2 in rects[n+1:]:
            if k1 == k2:
                continue
            if r1[0] < r2[1] and r2[0] < r1[1] and \
                    r1[2] < r2[3] and r2[2] < r1[3]:
                if (k1, k2) not in pairs:
                    pairs.append((k1, k2))
    return pairs

def rama_populations(phi, psi, bounds):
    """ Fraction of (phi, psi) points falling in each state.

    Counts direct assignments only, before unassigned points
    inherit the previous state. Overlapping regions go to the
    first state in ``bounds``, as in ``discrete_rama``.

    Parameters
    ----------
    phi, psi : array
        Angles in radians, any shape (all residues are pooled).
    bounds : dict
        State bounds in degrees.

    Returns
    -------
    pops : dict
        Fraction per state, plus 'O' for unassigned points.

    """
    f = np.degrees(np.ravel(phi))
    y = np.degrees(np.ravel(psi))
    free = np.ones(len(f), dtype=bool)
    pops = {}
    for k, b in bounds.items():
        inside = np.zeros(len(f), dtype=bool)
        for i in range(0, len(b), 4):
            inside |= (f > b[i]) & (f < b[i+1]) & \
                    (y > b[i+2]) & (y < b[i+3])
        inside &= free
        pops[k] = inside.mean() if len(f) else 0.
        free &= ~inside
    pops['O'] = free.mean() if len(f) else 0.
    return pops

def plot_rama_bounds(phi, psi, bounds, ax=None, bins=90):
    """ Ramachandran histogram with the state bounds drawn on top.

    Parameters
    ----------
    phi, psi : array
        Angles in radians, any shape (all residues are pooled).
    bounds : dict
        State bounds in degrees.
    ax : matplotlib Axes, optional
        Axes to draw on. A new figure is made by default.
    bins : int
        Number of histogram bins per axis.

    Returns
    -------
    ax : matplotlib Axes

    """
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    f = np.degrees(np.ravel(phi))
    y = np.degrees(np.ravel(psi))
    h, xe, ye = np.histogram2d(f, y, bins=bins, \
            range=[[-180, 180], [-180, 180]], density=True)
    h = np.ma.masked_where(h == 0, h)
    ax.pcolormesh(xe, ye, h.T, norm=LogNorm(), cmap='Greys')
    pops = rama_populations(phi, psi, bounds)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for n, (k, b) in enumerate(bounds.items()):
        c = colors[n % len(colors)]
        for i in range(0, len(b), 4):
            ax.add_patch(Rectangle((b[i], b[i+2]), b[i+1] - b[i], \
                    b[i+3] - b[i+2], fill=False, ec=c, lw=2, \
                    label="%s (%.1f%%)"%(k, 100*pops[k]) \
                    if i == 0 else None))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.set_xticks(range(-180, 181, 90))
    ax.set_yticks(range(-180, 181, 90))
    ax.set_xlabel(r'$\phi$ (deg)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.set_title('unassigned: %.1f%%'%(100*pops['O']))
    ax.legend(loc='upper right', fontsize='small')
    ax.figure.tight_layout()
    return ax

def _interactive():
    """ Whether a user can answer input() prompts. """
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None and getattr(ip, 'kernel', None) is not None:
            return True
    except ImportError:
        pass
    return sys.stdin is not None and sys.stdin.isatty()

def _ask(prompt, input_fn):
    """ Prompt the user; None if nobody can answer. """
    try:
        return input_fn(prompt).strip()
    except (EOFError, OSError, NotImplementedError):
        # pytest capture, closed stdin, or a kernel without stdin
        # (e.g. notebooks executed by nbconvert)
        return None

def confirm_bounds(phi, psi, bounds, input_fn=None, plot=True):
    """ Show the Ramachandran bounds and let the user revise them.

    Draws the pooled Ramachandran histogram with each state's
    region, reports the populations, and asks whether to accept.
    On refusal, asks for new bounds state by state (blank keeps
    the current ones) and shows the result again. When no user
    can answer (scripts without a terminal, pytest, nbconvert),
    the bounds are accepted with a warning.

    Parameters
    ----------
    phi, psi : array
        Angles in radians, any shape (all residues are pooled).
    bounds : dict
        Proposed state bounds in degrees.
    input_fn : callable, optional
        Replacement for ``input`` (used for testing). When given,
        the session is assumed to be interactive.
    plot : bool
        Whether to draw the histogram (the populations are always
        printed).

    Returns
    -------
    bounds : dict
        The accepted bounds.

    """
    import warnings
    bounds = {k: _validate_bounds(b) for k, b in bounds.items()}
    if input_fn is None:
        input_fn = input
        interactive = _interactive()
    else:
        interactive = True

    while True:
        if plot:
            plot_rama_bounds(phi, psi, bounds)
            plt.show()
            plt.close()
        pops = rama_populations(phi, psi, bounds)
        for k, b in bounds.items():
            print("  %s: %s  (%.1f%%)"%(k, b, 100*pops[k]))
        print("  unassigned: %.1f%%"%(100*pops['O']))
        for k1, k2 in _overlaps(bounds):
            print("  warning: %s and %s overlap; points in both go "\
                    "to %s"%(k1, k2, k1))

        ans = _ask("Accept these bounds? [Y/n]: ", input_fn) \
                if interactive else None
        if ans is None:
            warnings.warn("no interactive input available; using "\
                    "Ramachandran bounds without confirmation: %s"\
                    %bounds, stacklevel=2)
            return bounds
        if ans.lower() in ('', 'y', 'yes'):
            return bounds
        if ans.lower() not in ('n', 'no'):
            print("Please answer y or n.")
            continue

        for k in bounds:
            while True:
                new = _ask("%s %s -> phi_min, phi_max, psi_min, "\
                        "psi_max [, more groups of 4] (blank keeps): "\
                        %(k, bounds[k]), input_fn)
                if not new:
                    break
                try:
                    bounds[k] = _validate_bounds( \
                            new.replace(',', ' ').split())
                    break
                except ValueError as e:
                    print("  invalid: %s"%e)

def discrete_rama(phi, psi, bounds=None, states=None):
    """ Assign Ramachandran angles to coarse states.

    Follows the Buchete-Hummer assignment procedure.

    Parameters
    ----------
    phi : array
        Phi angles, shape (n_frames, n_dihedrals), in radians.
    psi : array
        Psi angles, shape (n_frames, n_dihedrals), in radians.
    bounds : dict, optional
        Custom bounds dict. Default uses standard A/E/L regions.
    states : list, optional
        State labels to include. Defaults to all defined labels.

    Returns
    -------
    cstates : list
        Per-frame state string (one character per dihedral pair).

    References
    ----------
    N.-V. Buchete and G. Hummer, J. Phys. Chem. B (2008).

    """
    if bounds is None:
        bounds = default_bounds(states)

    ndih = phi.shape[1]
    if psi.shape[1] != ndih:
        raise ValueError("phi and psi have different numbers of dihedrals")

    cstates = []
    prev_s_string = ['O'] * ndih
    for f, y in zip(phi, psi):
        s_string = []
        for n in range(ndih):
            s, _ = _state(f[n] * 180. / math.pi, y[n] * 180. / math.pi, bounds)
            if s == 'O':
                s_string.append(prev_s_string[n])
            else:
                s_string.append(s)
        cstates.append(''.join(s_string))
        prev_s_string = s_string
    return cstates
