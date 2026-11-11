import unittest
import os
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from mastermsm.test.download_data import download_osf_alaTB, download_osf_ala5, _DATA_DIR

class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        download_osf_alaTB()

        top = os.path.join(_DATA_DIR, 'alaTB.gro')
        xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')

        self.ts = traj.TimeSeries(xtc=xtc, top=top)
        self.tss = [traj.TimeSeries(xtc=xtc, top=top) \
                for i in range(2)]

        self.dis = traj.TimeSeries(distraj=[0,1,1,0])
        self.dis_multi = traj.TimeSeries(distraj=[[0,1,1,0], \
                [1,0,0,1]])

#class TestMDTrajLib(unittest.TestCase):
#    def setUp(self):
#        download_osf_alaTB()
#        self.ts = traj.TimeSeries(top='test/data/alaTB.gro', \
#                 trajs=['test/data/alaTB.xtc'])
#        self.ts_multi = traj.TimeSeries(top='test/data/alaTB.gro', \
#                       trajs=['test/data/alaTB.xtc', \
#                       'test/data/alaTB.xtc'])
#        self.ts_dis = traj.TimeSeries(dtrajs=[0,1,1,0])
#
#
#    def test_inrange(self):
#        self.inrange = traj_lib._inrange(2, 1, 3)
#        self.assertEqual(self.inrange, 1)
#        self.inrange = traj_lib._inrange(0, 1, 2)
#        self.assertEqual(self.inrange, 0)
#        self.inrange = traj_lib._inrange(1, 1, 2)
#        self.assertEqual(self.inrange, 0)
#
#    def test_inbounds(self):
#        TBA_bounds = {}
#        TBA_bounds['A'] = [-100., -40., -50., -10.]
#        TBA_bounds['E'] = [-180., -40., 125., 165.]
#        TBA_bounds['L'] = [50., 100., -40., 70.0]
#
#    #   test in alpha helix
#        self.inbounds = traj_lib._inbounds(TBA_bounds['A'], -90, -40)
#        self.assertEqual(self.inbounds, 1)
#    #   test in beta-sheet
#        self.inbounds = traj_lib._inbounds(TBA_bounds['E'], -90, 140)
#        self.assertEqual(self.inbounds, 1)
#    #   test in left-handed alpha helix
#        self.inbounds = traj_lib._inbounds(TBA_bounds['L'], 70, 30)
#        self.assertEqual(self.inbounds, 1)
#    #   test when no conformation
#        self.inbounds = traj_lib._inbounds(TBA_bounds['A'], 0, 0)
#        self.assertEqual(self.inbounds, 0)
#
#
#    def test_state(self):
#        psi = [-30, 0, -40, 90, 140, 180]
#        phi = [60., 0, -90, -90, -90, -180]
#        states_test = ['L','O','A','O','E','O']
#        bounds = {}
#        bounds['A'] = [-100., -40., -50., -10.]
#        bounds['E'] = [-180., -40., 125., 165.]
#        bounds['L'] = [50., 100., -40., 70.0]
#
#        for ind in range(len(phi)):
#            result = traj_lib._state(phi[ind], psi[ind], bounds)
#            state = result[0]
#            self.assertEqual(state, states_test[ind], 'expected state %s but got %s'%(state,states_test[ind]))
#
#    def test_stategrid(self):
#        self.assertIsNotNone(traj_lib._stategrid(-180, -180, 20))
#        self.assertLess(traj_lib._stategrid(-180, 0, 20),400)
#        self.assertEqual(traj_lib._stategrid(0, 0, 20), 210)
#        self.assertEqual(traj_lib._stategrid(-180, 0, 100), 2186)
#
#    def test_discreterama(self):
#        mdt_test = self.tr.mdt
#
#        phi = md.compute_phi(mdt_test)
#        psi = md.compute_psi(mdt_test)
#        # print(psi)
#        # psi = ([ 6,  8, 14, 16], [-30, 0, -40, 90, 140, 180])
#        # phi = ([ 4,  6,  8, 14],[60., 0, -90, -90, -90, -180])
#        states = ['L','A','E']
#        discrete = traj_lib.discrete_rama(phi, psi, states=states)
#        unique_st = set(discrete)
#        for state in unique_st:
#            self.assertIn(state, ['O', 'A', 'E', 'L'])
#
#    def test_discreteramagrid(self):
#        mdt_test = self.tr.mdt
#
#        phi = md.compute_phi(mdt_test)
#        psi = md.compute_psi(mdt_test)
#        discrete = traj_lib.discrete_ramagrid(phi, psi, nbins=20)
#        min_ibin = min(discrete)
#        max_ibin = max(discrete)
#        self.assertLess(max_ibin,400)
#        self.assertGreaterEqual(min_ibin,0)

class TestMDtraj(unittest.TestCase):
    def setUp(self):
        download_osf_alaTB()
        self.gro = os.path.join(_DATA_DIR, 'alaTB.gro')
        self.xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
        self.mdtraj = traj_lib.load_mdtraj(xtc=self.xtc, top=self.gro)
        self.mdtrajs = [traj_lib.load_mdtraj(xtc=self.xtc, \
                top=self.gro) for i in range(2)]

    def test_traj(self):
        self.assertIsNotNone(self.mdtraj)
        self.assertEqual(self.mdtraj.n_atoms, 19)
        self.assertEqual(self.mdtraj.timestep, 5.)
        self.assertEqual(self.mdtraj.n_residues, 3)
        self.assertEqual(self.mdtraj.n_frames, 40001)

    def test_trajs(self):
        self.assertEqual(len(self.mdtrajs), 2)
        self.assertEqual(self.mdtrajs[0].n_atoms, 19)
        self.assertEqual(self.mdtrajs[0].timestep, 5.)
        self.assertEqual(self.mdtrajs[0].n_residues, 3)

    def test_load_mdtraj(self):
        self.assertIsNotNone(self.mdtraj)
        self.assertEqual(self.mdtraj.__module__, 'mdtraj.core.trajectory')
        self.assertEqual(hasattr(self.mdtraj, '__class__'), True)

#    def test_read_distraj(self):
#        self.assertIsNotNone(self.tr._read_distraj)
#        self.assertEqual(callable(self.tr._read_distraj), True)
#    #   read distraj from temp file
#        content = "0.0 A\n" \
#                  "1.0 E\n" \
#                  "2.0 L\n" \
#                  "3.0 O"
#        fn = 'temp.txt'
#        fd = open(fn, 'w+')
#
#        try:
#            fd.write(content)
#            fd.seek(0)
#            cstates, dt = self.tr._read_distraj(distraj=fd.name)
#            self.assertIsInstance(cstates, list)
#            self.assertEqual(len(cstates), len(content.split('\n')))
#            self.assertEqual(dt, 1.0)
#
#        finally:
#            fd.close()
#            os.remove(fd.name)
#    #   read distraj from array and custom timestamp
#        distraj_arr = content.split('\n')
#        cstates, dt = self.tr._read_distraj(distraj=distraj_arr, dt=2.0)
#        self.assertIsInstance(cstates,list)
#        self.assertEqual(len(cstates), len(content.split('\n')))
#        self.assertEqual(dt, 2.0)
#    #   read empty 'discrete' trajectory
#        cstates, dt = self.tr._read_distraj(distraj=[])
#        self.assertEqual(len(cstates), 0)
#        self.assertEqual(dt, 1.0)
#
#    def test_timeseries_init(self):
#        self.assertIsNotNone(self.ts)
#        traj = self.ts.trajs[0]
#        self.assertIsNotNone(traj.mdt)
#        self.assertEqual(hasattr(traj.mdt, '__class__'), True)
#        self.assertEqual(traj.mdt.__module__ , 'mdtraj.core.trajectory')
#        self.assertIsNotNone(self.tr.discretize)
#
#    def test_ts_discretize(self):
#        self.tr.discretize('rama', states=['A', 'E', 'L'])
#        self.assertIsNotNone(self.tr.distraj)
#        unique_states = sorted(set(self.tr.distraj))
#        self.assertListEqual(unique_states, ['A', 'E', 'L', 'O'])
#
#    def test_ts_find_keys(self):
#        self.assertIsNotNone(self.tr.find_keys)
#    #   test excluding state O (unassigned)
#        self.tr.distraj = ['O']*50000
#        for i in range(len(self.tr.distraj)):
#            self.tr.distraj[i] = np.random.choice(['A', 'E', 'L', 'O'])
#
#        self.tr.find_keys()
#        keys = self.tr.keys
#        self.assertEqual(len(set(keys)), len(keys))
#        self.assertEqual(len(keys), 3)
#        for key in keys:
#            self.assertIn(key,['A','E','L'])
#
#        del self.tr.distraj
#    #   test excluding state in alpha-h
#        self.tr.distraj = ['O'] * 50000
#        for i in range(len(self.tr.distraj)):
#            self.tr.distraj[i] = np.random.choice(['A', 'E', 'L', 'O'])
#
#        self.tr.find_keys(exclude=['A'])
#        keys = self.tr.keys
#        self.assertEqual(len(set(keys)),len(keys))
#        self.assertEqual(len(keys), 3)
#        for key in keys:
#            self.assertIn(key,['O','E','L'])
#
#    def test_gc(self):
#        self.tr.gc()
#        self.assertIs(hasattr(self.tr, 'mdt'), False)

class UseMDtraj(unittest.TestCase):
    def setUp(self):
        download_osf_alaTB()
        top = os.path.join(_DATA_DIR, 'alaTB.gro')
        xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
        self.traj = traj.TimeSeries(xtc=xtc, top=top)
        self.trajs = [traj.TimeSeries(xtc=xtc, top=top) \
                for i in range(2)]

    def test_atributes(self):
        tr = self.traj
        self.assertIsNotNone(tr.mdt)
        self.assertEqual(tr.mdt.n_atoms, 19)
        self.assertEqual(tr.mdt.n_frames, 40001)
        self.assertEqual(tr.mdt.n_residues, 3)

class TestFeaturizer_alaTB(unittest.TestCase):
    def setUp(self):
        download_osf_alaTB()
        top = os.path.join(_DATA_DIR, 'alaTB.gro')
        xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
        self.ts = traj.TimeSeries(xtc=xtc, top=top)
        self.ts2 = [traj.TimeSeries(xtc=xtc, top=top) \
                for i in range(2)]

    def test_create_featurizer(self):
        feat = traj.Featurizer(self.ts)
        self.assertEqual(feat.n_trajs, 1)
        feat = traj.Featurizer(self.ts2)
        self.assertEqual(feat.n_trajs, 2)

    def test_torsions(self):
        feat = traj.Featurizer(self.ts)
        feat.add_torsions(shift=False)
        self.assertEqual(np.shape(self.ts.features),\
                (40001, 2))

    def test_torsions_sincos(self):
        feat = traj.Featurizer(self.ts)
        feat.add_torsions(shift=False, sincos=True)
        self.assertEqual(np.shape(self.ts.features),\
                (40001, 4))

    def test_torsions_shift(self):
        feat = traj.Featurizer(self.ts)
        feat.add_torsions(shift=True)
        self.assertTrue(np.all(self.ts.features[:,1] > -2))
        self.assertTrue(np.all(self.ts.features[:,0] > -2))

class TestFeaturizer_ala5(unittest.TestCase):
    def setUp(self):
        download_osf_ala5()
        top = os.path.join(_DATA_DIR, 'ala5.gro')
        xtc = os.path.join(_DATA_DIR, 'ala5.xtc')
        self.ts = traj.TimeSeries(xtc=xtc, top=top)
        self.ts2 = [traj.TimeSeries(xtc=xtc, top=top) \
                for i in range(2)]

    def test_create_featurizer(self):
        feat = traj.Featurizer(self.ts)
        self.assertEqual(feat.n_trajs, 1)
        feat = traj.Featurizer(self.ts2)
        self.assertEqual(feat.n_trajs, 2)

    def test_torsions(self):
        feat = traj.Featurizer(self.ts)
        feat.add_torsions(shift=False)
        self.assertEqual(np.shape(self.ts.features),\
                (50001, 10))

    def test_torsions_sincos(self):
        feat = traj.Featurizer(self.ts)
        feat.add_torsions(shift=False, sincos=True)
        self.assertEqual(np.shape(self.ts.features),\
                (50001, 20))

    def test_contacts(self):
        feat = traj.Featurizer(self.ts)
        feat.add_contacts()
        self.assertEqual(np.shape(self.ts.features),\
                (50001, 3))

class TestDimRed_alaTB(unittest.TestCase):
    def setUp(self):
        download_osf_alaTB()
        top = os.path.join(_DATA_DIR, 'alaTB.gro')
        xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
        self.ts = traj.TimeSeries(xtc=xtc, top=top)
        feat = traj.Featurizer(self.ts)
        feat.add_torsions(shift=False)

    def test_whiten(self):
        dr = traj.DimRed(self.ts)
        dr.whiten()
        self.assertLess(np.abs(np.mean(self.ts.features[:,0])), 1e-6)
        self.assertLess(np.abs(np.mean(self.ts.features[:,1])), 1e-6)

class TestDimRed_TICA(unittest.TestCase):
    """ TICA checked against an analytic AR(1)/OU process """
    def _ou(self, n=200000, a=0.95, seed=0):
        rng = np.random.RandomState(seed)
        x = np.zeros((n, 2))
        s = np.sqrt(1 - a**2)
        for i in range(1, n):
            x[i] = a*x[i-1] + s*rng.randn(2)
        # mix slow and fast coordinates so TICA must unmix them
        return x

    def _ts(self, x):
        ts = traj.TimeSeries()
        ts.features = x
        ts.dt = 1.0
        return ts

    def test_timescale(self):
        # two independent AR(1): eigenvalues a1^lag and a2^lag
        rng = np.random.RandomState(1)
        n, a = 200000, np.array([0.99, 0.9])
        x = np.zeros((n, 2))
        e = rng.randn(n, 2)*np.sqrt(1 - a**2)
        for i in range(1, n):
            x[i] = a*x[i-1] + e[i]
        x = x.dot(np.array([[1., 0.5], [0.3, 1.]]))   # mix
        ts = self._ts(x)
        dr = traj.DimRed(ts)
        dr.doTICA(lag=5, n=2)
        np.testing.assert_allclose(dr.tica_eigenvalues, a**5, atol=0.02)
        np.testing.assert_allclose(dr.tica_timescales, \
                -5/np.log(a**5), rtol=0.15)
        self.assertEqual(ts.features.shape, (n, 2))

    def test_whitened_projection(self):
        ts = self._ts(self._ou(n=50000))
        dr = traj.DimRed(ts)
        dr.doTICA(lag=3, n=2)
        self.assertLess(np.abs(ts.features.mean(axis=0)).max(), 0.05)

    def test_multiple_trajs(self):
        tss = [self._ts(self._ou(n=20000, seed=s)) for s in (2, 3)]
        dr = traj.DimRed(tss)
        dr.doTICA(lag=2, n=1)
        for t in tss:
            self.assertEqual(t.features.shape, (20000, 1))

    def test_bad_lag(self):
        ts = self._ts(self._ou(n=100))
        dr = traj.DimRed(ts)
        with self.assertRaises(ValueError):
            dr.doTICA(lag=0)
        with self.assertRaises(ValueError):
            dr.doTICA(lag=100)

class TestFilterStates(unittest.TestCase):
    def test_fills_with_previous(self):
        out = traj_lib._filter_states([0, -1, -1, 1, -1, 2])
        self.assertEqual(out, [0, 0, 0, 1, 1, 2])

    def test_leading_unassigned_keeps_length(self):
        # frames before the first assignment inherit the first label
        out = traj_lib._filter_states(np.array([-1, -1, 3, 3, -1, 4]))
        self.assertEqual(out, [3, 3, 3, 3, 3, 4])

    def test_none_assigned(self):
        with self.assertRaises(ValueError):
            traj_lib._filter_states([-1, -1])

class TestProtocol(unittest.TestCase):
    """ Discretization protocols on synthetic and real data """
    def _two_well(self, n=20000, seed=0):
        rng = np.random.RandomState(seed)
        s = np.cumsum(rng.rand(n) < 0.002) % 2    # slow switching
        x = np.column_stack([3.*s - 1.5, np.zeros(n)]) \
                + 0.3*rng.randn(n, 2)
        ts = traj.TimeSeries()
        ts.features = x
        ts.dt = 1.0
        return ts, s

    def test_kmeans_recovers_wells_and_is_nondestructive(self):
        ts, s = self._two_well()
        x0 = ts.features.copy()
        spec = {'whiten': False, 'cluster': ('kmeans', {'k': 2, 'dim': 2})}
        res = traj.discretize_protocol(ts, spec, name='km', seed=0)
        np.testing.assert_array_equal(ts.features, x0)
        self.assertFalse(hasattr(ts, 'distraj'))
        self.assertEqual(res.n_states, 2)
        self.assertEqual(res.frac_assigned, 1.0)
        d = np.asarray(res.timeseries[0].distraj)
        agree = max(np.mean(d == s), np.mean(d != s))
        self.assertGreater(agree, 0.99)

    def test_seed_reproducible(self):
        ts, _ = self._two_well()
        spec = {'cluster': ('kmeans', {'k': 10, 'dim': 2})}
        a = traj.discretize_protocol(ts, spec, seed=3)
        b = traj.discretize_protocol(ts, spec, seed=3)
        np.testing.assert_array_equal(a.timeseries[0].distraj, \
                b.timeseries[0].distraj)

    def test_tica_then_kmeans_multiple_trajs(self):
        tss = [self._two_well(n=10000, seed=i)[0] for i in (1, 2)]
        spec = {'reduce': ('tica', {'lag': 10, 'n': 1}),
                'cluster': ('kmeans', {'k': 2, 'dim': 1})}
        res = traj.discretize_protocol(tss, spec, seed=0)
        for tr, new in zip(tss, res.timeseries):
            self.assertEqual(tr.features.shape, (10000, 2))
            self.assertEqual(new.features.shape, (10000, 1))
            self.assertEqual(len(new.distraj), 10000)

    def test_compare_protocols(self):
        ts, _ = self._two_well(n=5000)
        protocols = {
            'pca': {'reduce': ('pca', {'n': 2}),
                    'cluster': ('kmeans', {'k': 4, 'dim': 2})},
            'raw': {'whiten': False,
                    'cluster': ('kmeans', {'k': 2, 'dim': 2})}}
        out = traj.compare_protocols(ts, protocols, seed=0)
        self.assertEqual(sorted(out), ['pca', 'raw'])
        self.assertEqual(out['pca'].n_states, 4)
        self.assertEqual(out['raw'].n_states, 2)

    def test_bad_specs(self):
        ts, _ = self._two_well(n=100)
        for spec in [{}, {'rama': ['A'], 'cluster': 'kmeans'},
                     {'cluster': 'nope'},
                     {'reduce': 'nope', 'cluster': 'kmeans'},
                     {'cluster': 'kmeans', 'typo': 1}]:
            with self.assertRaises(ValueError):
                traj.discretize_protocol(ts, spec)

    def test_no_features(self):
        ts = traj.TimeSeries()
        with self.assertRaises(ValueError):
            traj.discretize_protocol(ts, {'cluster': 'kmeans'})

    def test_rama_matches_timeseries_discretize(self):
        download_osf_alaTB()
        top = os.path.join(_DATA_DIR, 'alaTB.gro')
        xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
        ts = traj.TimeSeries(xtc=xtc, top=top)
        res = traj.discretize_protocol(ts, {'rama': ['A', 'E'],
                                            'check': False})
        self.assertFalse(hasattr(ts, 'distraj'))
        ts.discretize(states=['A', 'E'], check=False)
        self.assertEqual(res.timeseries[0].distraj, ts.distraj)
        self.assertLessEqual(res.n_states, 2)

    def test_featurized_protocol_leaves_input_clean(self):
        download_osf_alaTB()
        top = os.path.join(_DATA_DIR, 'alaTB.gro')
        xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
        ts = traj.TimeSeries(xtc=xtc, top=top)
        spec = {'features': [('torsions', {'sincos': True})],
                'reduce': ('pca', {'n': 2}),
                'cluster': ('kmeans', {'k': 10, 'dim': 2})}
        res = traj.discretize_protocol(ts, spec, seed=0)
        self.assertFalse(hasattr(ts, 'features'))
        self.assertFalse(hasattr(ts, 'distraj'))
        self.assertEqual(len(res.timeseries[0].distraj), ts.mdt.n_frames)
        self.assertEqual(res.n_states, 10)

    def test_hdbscan_fraction_assigned(self):
        try:
            import hdbscan
        except ImportError:
            self.skipTest("hdbscan not installed")
        ts, _ = self._two_well(n=5000)
        spec = {'whiten': False,
                'cluster': ('hdbscan', {'mcs': 100, 'dim': 2})}
        res = traj.discretize_protocol(ts, spec)
        self.assertGreater(res.frac_assigned, 0.5)
        self.assertEqual(len(res.timeseries[0].distraj), 5000)


class TestRamaBounds(unittest.TestCase):
    """ Interactive confirmation of Ramachandran bounds. """
    def setUp(self):
        # (phi, psi) in degrees: two A points, one E, one L, one O
        pts = np.array([[-60., -40.], [-70., -30.], [-120., 150.],
                        [60., 40.], [0., 0.]])
        self.phi = np.radians(pts[:, :1])
        self.psi = np.radians(pts[:, 1:])

    @staticmethod
    def _answers(*answers):
        it = iter(answers)
        return lambda prompt: next(it)

    def test_default_bounds(self):
        b = traj_lib.default_bounds(['A', 'E'])
        self.assertEqual(sorted(b), ['A', 'E'])
        self.assertEqual(b['A'], [-130., -40., -50., 20.])
        b['A'][0] = 0.
        self.assertEqual(traj_lib.default_bounds(['A'])['A'][0], -130.)
        with self.assertRaises(ValueError):
            traj_lib.default_bounds(['X'])

    def test_validate_bounds(self):
        for bad in [[], [1, 2, 3], [10, -10, 0, 10],
                    [-200, 0, 0, 10], ['a', 1, 2, 3]]:
            with self.assertRaises(ValueError):
                traj_lib._validate_bounds(bad)
        self.assertEqual(traj_lib._validate_bounds(['1', 2, 3, 4]),
                         [1., 2., 3., 4.])

    def test_populations(self):
        pops = traj_lib.rama_populations(self.phi, self.psi,
                                         traj_lib.default_bounds())
        self.assertAlmostEqual(pops['A'], 0.4)
        self.assertAlmostEqual(pops['E'], 0.2)
        self.assertAlmostEqual(pops['L'], 0.2)
        self.assertAlmostEqual(pops['O'], 0.2)

    def test_overlaps(self):
        b = {'A': [-100, 0, -50, 0], 'B': [-50, 50, -10, 10]}
        self.assertEqual(traj_lib._overlaps(b), [('A', 'B')])
        self.assertEqual(traj_lib._overlaps(traj_lib.default_bounds()),
                         [])

    def test_accept(self):
        b0 = traj_lib.default_bounds(['A', 'E'])
        for ans in ['', 'y', 'YES']:
            b = traj_lib.confirm_bounds(self.phi, self.psi, b0,
                    input_fn=self._answers(ans), plot=False)
            self.assertEqual(b, b0)

    def test_redefine(self):
        b0 = traj_lib.default_bounds(['A', 'E'])
        # reject, give bad then good A, keep E, then accept
        b = traj_lib.confirm_bounds(self.phi, self.psi, b0,
                input_fn=self._answers('maybe', 'n', '10, -10, 0, 1',
                                       '-90 0 -60 0', '', 'y'),
                plot=False)
        self.assertEqual(b['A'], [-90., 0., -60., 0.])
        self.assertEqual(b['E'], b0['E'])
        self.assertEqual(b0['A'], [-130., -40., -50., 20.])

    def test_plot(self):
        from unittest import mock
        import matplotlib
        matplotlib.use('Agg')
        with mock.patch.object(traj_lib.plt, 'show') as show:
            b = traj_lib.confirm_bounds(self.phi, self.psi,
                    traj_lib.default_bounds(),
                    input_fn=self._answers('n', '', '', '', 'y'))
        self.assertEqual(show.call_count, 2)
        self.assertEqual(sorted(b), ['A', 'E', 'L'])

    def test_noninteractive_warns(self):
        from unittest import mock
        b0 = traj_lib.default_bounds(['A'])
        with mock.patch.object(traj_lib, '_interactive',
                               return_value=False):
            with self.assertWarns(UserWarning):
                b = traj_lib.confirm_bounds(self.phi, self.psi, b0,
                                            plot=False)
        self.assertEqual(b, b0)

    def test_no_stdin_warns(self):
        def no_stdin(prompt):
            raise EOFError
        with self.assertWarns(UserWarning):
            traj_lib.confirm_bounds(self.phi, self.psi,
                    traj_lib.default_bounds(), input_fn=no_stdin,
                    plot=False)

    def test_discretizer_custom_bounds(self):
        download_osf_alaTB()
        top = os.path.join(_DATA_DIR, 'alaTB.gro')
        xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
        trs = [traj.TimeSeries(xtc=xtc, top=top) for i in range(2)]
        bounds = {'A': [-180., 0., -100., 50.],
                  'E': [-180., 0., 100., 180.]}
        disc = traj.Discretizer(trs)
        disc.rama(bounds=bounds, check=False)
        phi, psi = traj_lib.compute_rama(trs[0])
        ref = traj_lib.discrete_rama(phi, psi, bounds=bounds)
        self.assertEqual(disc.bounds, bounds)
        for tr in trs:
            self.assertEqual(tr.bounds, bounds)
            self.assertEqual(tr.distraj, ref)
        res = traj.discretize_protocol(trs, {'rama': bounds,
                                             'check': False})
        self.assertEqual(res.bounds, bounds)
        self.assertEqual(res.timeseries[0].distraj, ref)
        with self.assertRaises(ValueError):
            traj.discretize_protocol(trs, {'cluster': 'kmeans',
                                           'check': False})


# ---------------------------------------------------------------------
# Tests for code added on develop since master (TICA, protocols,
# Ramachandran bounds), covering branches the tests above leave out.
# ---------------------------------------------------------------------
import io
import sys
import contextlib
from types import SimpleNamespace
from unittest import mock


def _quiet(f, *args, **kwargs):
    """ Call f with stdout silenced (doTICA and friends print). """
    with contextlib.redirect_stdout(io.StringIO()):
        return f(*args, **kwargs)


def _alaTB(n=1):
    download_osf_alaTB()
    top = os.path.join(_DATA_DIR, 'alaTB.gro')
    xtc = os.path.join(_DATA_DIR, 'alaTB.xtc')
    trs = [traj.TimeSeries(xtc=xtc, top=top) for i in range(n)]
    return trs[0] if n == 1 else trs


class TestDimRed_TICA_extra(unittest.TestCase):
    """ Input validation, estimator and output conventions of doTICA """
    def _ts(self, x, dt=None):
        ts = traj.TimeSeries()
        ts.features = x
        if dt is not None:
            ts.dt = dt
        return ts

    def _ar1(self, n=50000, a=(0.98, 0.8), seed=0):
        rng = np.random.RandomState(seed)
        a = np.asarray(a)
        x = np.zeros((n, len(a)))
        e = rng.randn(n, len(a))*np.sqrt(1 - a**2)
        for i in range(1, n):
            x[i] = a*x[i-1] + e[i]
        return x.dot(np.array([[1., 0.5], [0.3, 1.]]))

    def test_bad_lag_values(self):
        dr = traj.DimRed(self._ts(self._ar1(n=200)))
        for lag in [None, -1, 2.5]:
            with self.assertRaises(ValueError):
                _quiet(dr.doTICA, lag=lag)

    def test_bad_n(self):
        dr = traj.DimRed(self._ts(self._ar1(n=200)))
        for n in [0, 3]:
            with self.assertRaises(ValueError):
                _quiet(dr.doTICA, lag=1, n=n)

    def test_mismatched_features(self):
        rng = np.random.RandomState(0)
        tss = [self._ts(rng.randn(100, 2)), self._ts(rng.randn(100, 3))]
        with self.assertRaises(ValueError):
            _quiet(traj.DimRed(tss).doTICA, lag=1, n=1)

    def test_lag_longer_than_one_traj(self):
        rng = np.random.RandomState(0)
        tss = [self._ts(rng.randn(100, 2)), self._ts(rng.randn(10, 2))]
        with self.assertRaises(ValueError):
            _quiet(traj.DimRed(tss).doTICA, lag=10, n=1)

    def test_constant_feature_raises(self):
        rng = np.random.RandomState(0)
        x = np.column_stack([rng.randn(1000), np.ones(1000)])
        with self.assertRaises(ValueError):
            _quiet(traj.DimRed(self._ts(x)).doTICA, lag=1, n=1)

    def test_float_constant_feature_raises(self):
        # mean of a non-representable constant leaves a ~1e-17 spread
        rng = np.random.RandomState(0)
        x = np.column_stack([rng.randn(1000), np.full(1000, 0.1)])
        with self.assertRaises(ValueError):
            _quiet(traj.DimRed(self._ts(x)).doTICA, lag=1, n=1)

    def test_feature_units_do_not_matter(self):
        # very different scales must not look singular
        x = self._ar1()*np.array([1e6, 1e-4])
        dr = traj.DimRed(self._ts(x))
        _quiet(dr.doTICA, lag=5, n=2)
        dr1 = traj.DimRed(self._ts(self._ar1()))
        _quiet(dr1.doTICA, lag=5, n=2)
        np.testing.assert_allclose(dr.tica_eigenvalues, \
                dr1.tica_eigenvalues, rtol=1e-6)

    def test_collinear_features_raise(self):
        # an exact linear combination makes C(0) singular; roundoff
        # can let the Cholesky step in eigh succeed, so doTICA must
        # check explicitly (it used to pass for some seeds)
        for seed in range(20):
            rng = np.random.RandomState(seed)
            b = rng.randn(5000, 2)
            x = np.column_stack([b, b[:, 0] + 2*b[:, 1]])
            with self.assertRaises(ValueError):
                _quiet(traj.DimRed(self._ts(x)).doTICA, lag=1, n=1)

    def test_projection_and_normalization(self):
        x = self._ar1()
        ts = self._ts(x.copy())
        dr = traj.DimRed(ts)
        _quiet(dr.doTICA, lag=5, n=2)
        # stored eigenvectors reproduce the projection
        np.testing.assert_allclose(ts.features, \
                (x - dr.tica_mean).dot(dr.tica_eigenvectors[:, :2]))
        # components are C(0)-orthonormal: unit variance, uncorrelated
        np.testing.assert_allclose(np.cov(ts.features.T), np.eye(2), \
                atol=0.02)
        ev = dr.tica_eigenvalues
        self.assertTrue(np.all(np.diff(ev) <= 0))
        self.assertTrue(np.all(ev <= 1 + 1e-10))

    def test_timescale_units(self):
        x = self._ar1()
        out = {}
        for dt in [None, 1.0, 2.0]:
            dr = traj.DimRed(self._ts(x.copy(), dt=dt))
            _quiet(dr.doTICA, lag=5, n=2)
            out[dt] = dr.tica_timescales
        # without dt, timescales are in frames (same as dt = 1)
        np.testing.assert_allclose(out[None], out[1.0])
        np.testing.assert_allclose(out[2.0], 2*out[1.0])

    def test_nonpositive_eigenvalues_give_nan(self):
        # white noise: about half the eigenvalues are negative
        rng = np.random.RandomState(0)
        dr = traj.DimRed(self._ts(rng.randn(5000, 4)))
        _quiet(dr.doTICA, lag=1, n=4)
        ev, ts = dr.tica_eigenvalues, dr.tica_timescales
        self.assertTrue(np.any(ev <= 0))
        self.assertTrue(np.all(np.isnan(ts[ev <= 0])))
        self.assertTrue(np.all(np.isfinite(ts[(ev > 0) & (ev < 1)])))

    def test_pairs_do_not_straddle_trajectories(self):
        # reference estimator built pair by pair, per trajectory
        rng = np.random.RandomState(4)
        xs = [rng.randn(60, 2) + 3., rng.randn(90, 2) - 3.]
        lag = 3
        x0 = np.array([x[t] for x in xs for t in range(len(x) - lag)])
        xt = np.array([x[t+lag] for x in xs \
                for t in range(len(x) - lag)])
        m = 0.5*(x0.mean(0) + xt.mean(0))
        x0, xt = x0 - m, xt - m
        c0 = 0.5*(x0.T.dot(x0) + xt.T.dot(xt))/len(x0)
        ct = x0.T.dot(xt)/len(x0)
        ct = 0.5*(ct + ct.T)
        ref = np.sort(np.linalg.eigvals(np.linalg.solve(c0, ct)).real)
        dr = traj.DimRed([self._ts(x.copy()) for x in xs])
        _quiet(dr.doTICA, lag=lag, n=2)
        np.testing.assert_allclose(np.sort(dr.tica_eigenvalues), ref)
        # a single concatenated trajectory gives a different answer
        dr1 = traj.DimRed(self._ts(np.vstack(xs)))
        _quiet(dr1.doTICA, lag=lag, n=2)
        self.assertFalse(np.allclose(np.sort(dr1.tica_eigenvalues), ref))


class TestDiscretizer_extra(unittest.TestCase):
    def _ts(self, n=3000, seed=0):
        rng = np.random.RandomState(seed)
        ts = traj.TimeSeries()
        ts.features = np.vstack([rng.randn(n//2, 2) - 3, \
                rng.randn(n - n//2, 2) + 3])
        return ts

    def test_single_timeseries_is_wrapped(self):
        ts = self._ts()
        self.assertEqual(traj.Discretizer(ts).timeseries, [ts])

    def test_kmeans_random_state(self):
        runs = []
        for i in range(2):
            ts = self._ts()
            disc = traj.Discretizer(ts)
            disc.kmeans(k=8, dim=2, random_state=7)
            runs.append((np.asarray(ts.distraj), disc.k_centers))
        np.testing.assert_array_equal(runs[0][0], runs[1][0])
        # centers agree to roundoff only (threaded reductions)
        np.testing.assert_allclose(runs[0][1], runs[1][1], atol=1e-12)

    def test_hdbscan_missing(self):
        disc = traj.Discretizer(self._ts())
        with mock.patch.dict(sys.modules, {'hdbscan': None}):
            with self.assertRaises(ImportError):
                disc.hdbscan(mcs=50, dim=2)

    def test_hdbscan_labels_filled(self):
        try:
            import hdbscan
        except ImportError:
            self.skipTest("hdbscan not installed")
        tss = [self._ts(seed=s) for s in (0, 1)]
        disc = traj.Discretizer(tss)
        disc.hdbscan(mcs=100, dim=2)
        self.assertEqual(disc.hdb_nclusters, 2)
        self.assertTrue(0 < disc.frac_assigned <= 1)
        for ts in tss:
            self.assertEqual(len(ts.distraj), 3000)
            self.assertGreaterEqual(min(ts.distraj), 0)


class TestProtocol_extra(unittest.TestCase):
    def _ts(self, n=2000, seed=0):
        rng = np.random.RandomState(seed)
        ts = traj.TimeSeries()
        ts.features = rng.randn(n, 2)
        return ts

    def test_repr_and_default_names(self):
        ts = self._ts()
        res = traj.discretize_protocol(ts, {'cluster': ('kmeans', \
                {'k': 3, 'dim': 2})}, seed=0)
        self.assertEqual(res.name, 'protocol')
        self.assertEqual(repr(res), \
                "Discretization('protocol', n_states=3, "\
                "frac_assigned=1.000)")

    def test_seed_does_not_mutate_spec(self):
        kw = {'k': 3, 'dim': 2}
        spec = {'cluster': ('kmeans', kw)}
        res = traj.discretize_protocol(self._ts(), spec, seed=1)
        self.assertEqual(kw, {'k': 3, 'dim': 2})
        self.assertEqual(res.spec, spec)

    def test_unknown_feature(self):
        ts = _alaTB()
        with self.assertRaises(ValueError):
            traj.discretize_protocol(ts, {'features': [('nope', {})], \
                    'cluster': 'kmeans'})

    def test_features_spec_replaces_existing_features(self):
        ts = _alaTB()
        traj.Featurizer(ts).add_torsions(shift=False)
        x0 = ts.features.copy()
        spec = {'features': [('torsions', {'sincos': True})],
                'cluster': ('kmeans', {'k': 5, 'dim': 4})}
        res = traj.discretize_protocol(ts, spec, seed=0)
        self.assertEqual(res.timeseries[0].features.shape, \
                (ts.mdt.n_frames, 4))
        np.testing.assert_array_equal(ts.features, x0)

    def test_existing_features_are_copied(self):
        ts = self._ts()
        x0 = ts.features.copy()
        res = traj.discretize_protocol(ts, {'whiten': False, \
                'cluster': ('kmeans', {'k': 2, 'dim': 2})}, seed=0)
        new = res.timeseries[0]
        self.assertIsNot(new.features, ts.features)
        np.testing.assert_array_equal(new.features, x0)

    def test_compare_protocols_independent(self):
        ts = self._ts()
        x0 = ts.features.copy()
        spec = {'cluster': ('kmeans', {'k': 4, 'dim': 2})}
        out = traj.compare_protocols(ts, {'a': spec, 'b': spec}, seed=0)
        self.assertEqual(out['a'].name, 'a')
        self.assertIsNot(out['a'].timeseries[0], out['b'].timeseries[0])
        np.testing.assert_array_equal(out['a'].timeseries[0].distraj, \
                out['b'].timeseries[0].distraj)
        np.testing.assert_array_equal(ts.features, x0)
        self.assertFalse(hasattr(ts, 'distraj'))


class TestRama_extra(unittest.TestCase):
    """ State assignment, bounds checks and user interaction """
    def test_discrete_rama_states_and_shapes(self):
        # one dihedral: A, unassigned (inherits A), E
        phi = np.radians([[-60.], [0.], [-120.]])
        psi = np.radians([[-40.], [0.], [150.]])
        self.assertEqual(traj_lib.discrete_rama(phi, psi, \
                states=['A', 'E']), ['A', 'A', 'E'])
        # without E, the last frame is unassigned and inherits A
        self.assertEqual(traj_lib.discrete_rama(phi, psi, \
                states=['A']), ['A', 'A', 'A'])
        with self.assertRaises(ValueError):
            traj_lib.discrete_rama(phi, np.zeros((3, 2)))

    def test_multi_region_bounds(self):
        b = {'A': [-100., -50., -60., -20., 50., 100., 20., 60.]}
        phi = np.radians([[-70.], [70.], [0.]])
        psi = np.radians([[-40.], [40.], [0.]])
        self.assertEqual(traj_lib.discrete_rama(phi, psi, bounds=b), \
                ['A', 'A', 'A'])
        pops = traj_lib.rama_populations(phi, psi, b)
        self.assertAlmostEqual(pops['A'], 2/3.)
        self.assertAlmostEqual(pops['O'], 1/3.)

    def test_populations_match_state_assignment(self):
        # overlapping and multi-region bounds, order matters
        b = {'A': [-130., -40., -50., 20.],
             'B': [-90., 0., -20., 40., 30., 100., -40., 70.],
             'E': [-180., -40., 125., 180.]}
        rng = np.random.RandomState(0)
        f = rng.uniform(-180, 180, 5000)
        y = rng.uniform(-180, 180, 5000)
        # include points exactly on edges (open intervals)
        f[:3], y[:3] = [-40., -130., 0.], [-30., -50., 0.]
        labels = [traj_lib._state(a, c, b)[0] for a, c in zip(f, y)]
        pops = traj_lib.rama_populations(np.radians(f), \
                np.radians(y), b)
        for k in ['A', 'B', 'E', 'O']:
            self.assertAlmostEqual(pops[k], labels.count(k)/5000.)
        self.assertEqual(labels[:3], ['O', 'O', 'O'])

    def test_overlaps_multi_region(self):
        # A's two rectangles overlap each other (same state: ignored);
        # A's second rectangle overlaps B
        b = {'A': [-100, 0, -50, 0, -50, 50, -20, 10],
             'B': [40, 90, 0, 50]}
        self.assertEqual(traj_lib._overlaps(b), [('A', 'B')])
        self.assertEqual(traj_lib._overlaps( \
                {'A': [-100, 0, -50, 0, -50, 50, -20, 10]}), [])

    def test_confirm_reports_overlap(self):
        b = {'A': [-100, 0, -50, 0], 'B': [-50, 50, -10, 10]}
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            traj_lib.confirm_bounds(np.zeros(3), np.zeros(3), b, \
                    input_fn=lambda p: 'y', plot=False)
        self.assertIn("A and B overlap", out.getvalue())

    def test_interactive_kernel(self):
        try:
            import IPython
        except ImportError:
            self.skipTest("IPython not installed")
        with mock.patch('IPython.get_ipython', \
                return_value=SimpleNamespace(kernel=object())):
            with mock.patch.object(traj_lib.sys, 'stdin', None):
                self.assertTrue(traj_lib._interactive())

    def test_interactive_stdin(self):
        tty = mock.Mock(isatty=lambda: True)
        pipe = mock.Mock(isatty=lambda: False)
        # no IPython at all, and IPython without a kernel
        for mods in [{'IPython': None}, {}]:
            with mock.patch.dict(sys.modules, mods):
                with mock.patch('IPython.get_ipython', \
                        return_value=None, create=True) \
                        if not mods else contextlib.nullcontext():
                    for stdin, expect in [(tty, True), (pipe, False), \
                            (None, False)]:
                        with mock.patch.object(traj_lib.sys, 'stdin', \
                                stdin):
                            self.assertEqual(traj_lib._interactive(), \
                                    expect)

    def test_timeseries_discretize_uses_confirmed_bounds(self):
        ts = _alaTB()
        new = {'A': [-180., 0., -100., 50.]}
        with mock.patch.object(traj_lib, 'confirm_bounds', \
                return_value=new) as cb:
            ts.discretize(states=['A', 'E'])
        cb.assert_called_once()
        self.assertEqual(cb.call_args[0][2], \
                traj_lib.default_bounds(['A', 'E']))
        self.assertEqual(ts.bounds, new)
        phi, psi = traj_lib.compute_rama(ts)
        self.assertEqual(ts.distraj, \
                traj_lib.discrete_rama(phi, psi, bounds=new))

    def test_discretizer_rama_pools_trajectories(self):
        trs = _alaTB(n=2)
        new = {'E': [-180., 0., 100., 180.]}
        with mock.patch.object(traj_lib, 'confirm_bounds', \
                return_value=new) as cb:
            disc = traj.Discretizer(trs)
            disc.rama()
        cb.assert_called_once()
        phi, psi, b0 = cb.call_args[0]
        n = sum(tr.mdt.n_frames for tr in trs)
        self.assertEqual(len(phi), n)
        self.assertEqual(len(psi), n)
        self.assertEqual(b0, traj_lib.default_bounds())
        self.assertEqual(disc.bounds, new)
        for tr in trs:
            self.assertEqual(tr.bounds, new)
            self.assertEqual(set(''.join(tr.distraj)) - {'O'}, {'E'})

    def test_protocol_rama_checks_by_default(self):
        ts = _alaTB()
        new = {'A': [-180., 0., -100., 50.]}
        with mock.patch.object(traj_lib, 'confirm_bounds', \
                return_value=new) as cb:
            res = traj.discretize_protocol(ts, {'rama': ['A']})
        cb.assert_called_once()
        self.assertEqual(res.bounds, new)
        self.assertFalse(hasattr(ts, 'bounds'))
