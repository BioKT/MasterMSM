import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from test.download_data import download_osf_alaTB
from test.download_data import download_osf_ala5
import os

class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        download_osf_alaTB()

        top = 'test/data/alaTB.gro'
        xtc = 'test/data/alaTB.xtc'

        self.ts = traj.TimeSeries(xtc=xtc, top=top)
        self.tss = [traj.TimeSeries(xtc=xtc, top=top) \
                for i in range(2)]

        self.dis = traj.TimeSeries(dtraj=[0,1,1,0])
        self.dis_multi = traj.TimeSeries(dtraj=[[0,1,1,0], \
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
        self.gro = 'test/data/alaTB.gro'
        self.xtc = 'test/data/alaTB.xtc'
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
        top = 'test/data/alaTB.gro'
        xtc = 'test/data/alaTB.xtc'
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
        top = 'test/data/alaTB.gro'
        xtc = 'test/data/alaTB.xtc'
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

    def test_torsions_shift(self):
        feat = traj.Featurizer(self.ts)
        feat.add_torsions(shift=True)
        self.assertTrue(np.all(self.ts.features[:,1] > -2))
        self.assertTrue(np.all(self.ts.features[:,0] > -2))

class TestFeaturizer_ala5(unittest.TestCase):
    def setUp(self):
        download_osf_ala5()
        top = 'test/data/ala5.gro'
        xtc = 'test/data/ala5.xtc'
        self.ts = traj.TimeSeries(xtc=xtc, top=top)
        self.ts2 = [traj.TimeSeries(xtc=xtc, top=top) \
                for i in range(2)]

    def test_create_featurizer(self):
        feat = traj.Featurizer(self.ts)
        self.assertEqual(feat.n_trajs, 1)
        feat = traj.Featurizer(self.ts2)
        self.assertEqual(feat.n_trajs, 2)
#
#    def test_torsions(self):
#        feat = traj.Featurizer(self.ts)
#        feat.add_torsions(shift=False)
#        self.assertEqual(np.shape(datasets.trajs[0].features),\
#                (10003, 2))
#        disc.add_torsions(shift=True)
#        self.assertTrue(np.all(datasets.trajs[0].features[:,1] > -2))
#        self.assertTrue(np.all(datasets.trajs[0].features[:,0] > -2))
#
#    def test_contacts(self):
#        feat = traj.Featurizer(self.ts)
#        feat.add_contacts()
#        self.assertEqual(np.shape(datasets.trajs[0].features),\
#                (10003, 2))
#        disc.add_torsions(shift=True)
#        self.assertTrue(np.all(datasets.trajs[0].features[:,1] > -2))
#        self.assertTrue(np.all(datasets.trajs[0].features[:,0] > -2))
