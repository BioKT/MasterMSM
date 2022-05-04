import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from test.download_data import download_test_data
import os

class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        download_test_data()
        self.xtc = traj.TimeSeries(top='test/data/alaTB.gro', \
                    trajs=['test/data/protein_only.xtc'])
        self.xtc_multi = traj.TimeSeries(top='test/data/alaTB.gro', \
                       trajs=['test/data/protein_only.xtc', \
                       'test/data/protein_only.xtc'])

        self.dis = traj.TimeSeries(dtrajs=[0,1,1,0])
        self.dis_multi = traj.TimeSeries(dtrajs=[[0,1,1,0], \
                [1,0,0,1]])

    def test_timeseries(self):
        self.assertEqual(self.xtc.n_trajs, 1)
        self.assertEqual(self.xtc_multi.n_trajs, 2)
        self.assertEqual(self.dis.n_trajs, 1)
        self.assertEqual(self.dis_multi.n_trajs, 2)

#class TestMDTrajLib(unittest.TestCase):
#    def setUp(self):
#        download_test_data()
#        self.xtc = traj.TimeSeries(top='test/data/alaTB.gro', \
#                    trajs=['test/data/protein_only.xtc'])
#        self.xtc_multi = traj.TimeSeries(top='test/data/alaTB.gro', \
#                       trajs=['test/data/protein_only.xtc', \
#                       'test/data/protein_only.xtc'])
#        self.ts_dis = traj.TimeSeries(dtrajs=[0,1,1,0])


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
        download_test_data()
        self.traj = md.load('test/data/protein_only.xtc', \
                top='test/data/alaTB.gro')
        self.topfn = 'test/data/alaTB.gro'
        self.trajfn = 'test/data/protein_only.xtc'
        self.ts = traj.TimeSeries(top='test/data/alaTB.gro', \
                             trajs=['test/data/protein_only.xtc'])

    def test_traj(self):
        self.assertIsNotNone(self.traj)
        self.assertEqual(self.traj.n_atoms, 19)
        self.assertEqual(self.traj.timestep, 1.)
        self.assertEqual(self.traj.n_residues, 3)
        self.assertEqual(self.traj.n_frames, 10003)

    def test_load_mdtraj(self):
        mdtraj = traj_lib.load_mdtraj(top=self.topfn, traj=self.trajfn)
        self.assertIsNotNone(mdtraj)
        self.assertEqual(mdtraj.__module__, 'mdtraj.core.trajectory')
        self.assertEqual(hasattr(mdtraj, '__class__'), True)

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
    def test_timeseries_init(self):
        self.assertIsNotNone(self.ts)
        traj = self.ts.trajs[0]
        self.assertIsNotNone(traj.mdt)
        self.assertEqual(hasattr(traj.mdt, '__class__'), True)
        self.assertEqual(traj.mdt.__module__ , 'mdtraj.core.trajectory')
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
        download_test_data()
        self.ts = traj.TimeSeries(top='test/data/alaTB.gro', \
                trajs=['test/data/protein_only.xtc'])

    def test_atributes(self):
        tr = self.ts.trajs[0]
        self.assertIsNotNone(tr.mdt)
        self.assertEqual(tr.mdt.n_atoms, 19)
        self.assertEqual(tr.mdt.n_frames, 10003)
        self.assertEqual(tr.mdt.n_residues, 3)
#        self.assertIsNotNone(self.tr.discretize)
#        self.assertIs(callable(self.tr.discretize), True)

class TestDiscretizer(object):
    def setUp(self):
        download_test_data()
        self.ts = traj.TimeSeries(top='test/data/alaTB.gro', \
                trajs=['test/data/protein_only.xtc'])

    def test_create_discretizer(self):
        disc = traj.Discretizer(self.ts)
        self.assertEqual(disc.timeseries.n_trajs, 1)

    def test_torsions(self):
        disc = traj.Discretizer(self.ts)
        disc.add_torsions()
        self.assertEqual(np.shape(datasets.trajs[0].feature_vector),\
                (10003, 2))
