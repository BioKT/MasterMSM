import unittest
import mdtraj as md
import numpy as np
from mastermsm.trajectory import traj_lib, traj
from mastermsm.msm import msm, msm_lib
import os
import matplotlib.pyplot as plt
class TestMDTrajLib(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                                  traj=['trajectory/tests/data/protein_only.xtc'])
    def test_inrange(self):
        assert traj_lib._inrange(2, 1, 3) == 1
        assert traj_lib._inrange(0, 1, 2) == 0
        assert traj_lib._inrange(1, 1, 2) == 0
    def test_inbounds(self):
        TBA_bounds = {}
        TBA_bounds['A'] = [-100., -40., -50., -10.]
        TBA_bounds['E'] = [-180., -40., 125., 165.]
        TBA_bounds['L'] = [50., 100., -40., 70.0]

    #   test in alpha helix
        assert traj_lib._inbounds(TBA_bounds['A'], -90, -40) == 1
    #   test in beta-sheet
        assert traj_lib._inbounds(TBA_bounds['E'], -90, 140) == 1
    #   test in left-handed alpha helix
        assert traj_lib._inbounds(TBA_bounds['L'], 70, 30) == 1
    #   test when no conformation
        assert traj_lib._inbounds(TBA_bounds['A'], 0, 0) == 0
    #   include more limit params

    def test_state(self):
        psi = [-30, 0, -40, 90, 140, 180]
        phi = [60., 0, -90, -90, -90, -180]
        states_test = ['L','O','A','O','E','O']
        bounds = {}
        bounds['A'] = [-100., -40., -50., -10.]
        bounds['E'] = [-180., -40., 125., 165.]
        bounds['L'] = [50., 100., -40., 70.0]

        for ind in range(len(phi)):
            result, angles = traj_lib._state(phi[ind], psi[ind], bounds)
            state = result[0]
            self.assertEqual(state,states_test[ind],'expected state %s but got %s'%(state,states_test[ind]))

    def test_stategrid(self):
        self.assertIsNotNone(traj_lib._stategrid(-180, -180, 20))
        self.assertLess(traj_lib._stategrid(-180, 0, 20),400)
        self.assertEqual(traj_lib._stategrid(0, 0, 20), 210)
        self.assertEqual(traj_lib._stategrid(-180, 0, 100), 2186)
    def test_discreterama(self):
        mdt_test = self.tr.mdt

        phi = md.compute_phi(mdt_test)
        psi = md.compute_psi(mdt_test)
        # print(psi)
        # psi = ([ 6,  8, 14, 16], [-30, 0, -40, 90, 140, 180])
        # phi = ([ 4,  6,  8, 14],[60., 0, -90, -90, -90, -180])
        states = ['L','A','E']
        discrete = traj_lib.discrete_rama(phi, psi, states=states)
        unique_st = set(discrete)
        for state in unique_st:
            self.assertIn(state, ['O', 'A', 'E', 'L'])
    def test_discreteramagrid(self):
        mdt_test = self.tr.mdt

        phi = md.compute_phi(mdt_test)
        psi = md.compute_psi(mdt_test)
        discrete = traj_lib.discrete_ramagrid(phi, psi, nbins=20)
        min_ibin = min(discrete)
        max_ibin = max(discrete)
        self.assertLess(max_ibin,400)
        self.assertGreaterEqual(min_ibin,0)

class TestMDtraj(unittest.TestCase):
    def setUp(self):
        self.traj = md.load('trajectory/tests/data/protein_only.xtc', \
                top='trajectory/tests/data/alaTB.gro')
        self.topfn = 'trajectory/tests/data/alaTB.gro'
        self.trajfn = 'trajectory/tests/data/protein_only.xtc'
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                                  traj=['trajectory/tests/data/protein_only.xtc'])

    def test_traj(self):
        assert self.traj is not None
        assert self.traj.n_atoms == 19
        assert self.traj.timestep == 1.0
        assert self.traj.n_residues == 3
        assert self.traj.n_frames == 10003

    def test_load_mdtraj(self):
        mdtraj = traj._load_mdtraj(top=self.topfn, traj=self.trajfn)
        assert mdtraj is not None
        assert mdtraj.__module__ == 'mdtraj.core.trajectory'
        assert hasattr(mdtraj, '__class__')

    def test_read_distraj(self):
        assert self.tr._read_distraj is not None
        assert callable(self.tr._read_distraj) is True
    #   read distraj from temp file
        content = "0.0 A\n" \
                  "1.0 E\n" \
                  "2.0 L\n" \
                  "3.0 O"
        fn = 'temp.txt'
        fd = open(fn, 'w+')

        try:
            fd.write(content)
            fd.seek(0)
            cstates, dt = self.tr._read_distraj(distraj=fd.name)
            assert isinstance(cstates, list)
            assert len(cstates) == len(content.split('\n'))
            assert dt == 1.0

        finally:
            fd.close()
            os.remove(fd.name)
    #   read distraj from array and custom timestamp
        distraj_arr = content.split('\n')
        cstates, dt = self.tr._read_distraj(distraj=distraj_arr, dt=2.0)
        assert isinstance(cstates, list)
        assert len(cstates) == len(content.split('\n'))
        assert dt == 2.0
    #   read empty 'discrete' trajectory
        cstates, dt = self.tr._read_distraj(distraj=[])
        assert len(cstates) == 0
        assert dt == 1.0

    def test_timeseries_init(self):
        assert self.tr is not None
        assert self.tr.mdt is not None
        assert hasattr(self.tr.mdt, '__class__')
        assert self.tr.mdt.__module__ == 'mdtraj.core.trajectory'
        assert self.tr.discretize is not None

    def test_ts_discretize(self):
        self.tr.discretize('rama', states=['A', 'E', 'L'])
        assert self.tr.distraj is not None
        unique_states = sorted(set(self.tr.distraj))
        assert unique_states == ['A', 'E', 'L', 'O']

    def test_ts_find_keys(self):
        assert self.tr.find_keys is not None
        assert hasattr(self.tr, 'find_keys') is True
    #   test excluding state O (unassigned)
        self.tr.distraj = ['O']*50000
        for i in range(len(self.tr.distraj)):
            self.tr.distraj[i] = np.random.choice(['A', 'E', 'L', 'O'])

        self.tr.find_keys()
        keys = self.tr.keys
        assert (len(set(keys)) == len(keys))
        for key in keys:
            assert (key in ['A', 'E', 'L'] and len(keys) == 3)
        del self.tr.distraj
    #   test excluding state in alpha-h
        self.tr.distraj = ['O'] * 50000
        for i in range(len(self.tr.distraj)):
            self.tr.distraj[i] = np.random.choice(['A', 'E', 'L', 'O'])

        self.tr.find_keys(exclude=['A'])
        keys = self.tr.keys
        assert (len(set(keys)) == len(keys))
        assert len(keys) == 3
        for key in keys:
            assert (key in ['E', 'L', 'O'])

    def test_gc(self):
        self.tr.gc()
        assert hasattr(self.tr, 'mdt') is False


class UseMDtraj(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                traj=['trajectory/tests/data/protein_only.xtc'])

    def test_atributes(self):
        assert self.tr.mdt is not None
        assert self.tr.mdt.n_atoms == 19
        assert self.tr.mdt.n_frames == 10003
        assert self.tr.mdt.n_residues == 3
        assert self.tr.discretize is not None
        assert callable(self.tr.discretize) is True


#    def test_discretize(self):
#        assert self.tr.n_traj == 1
class TestMSM(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                traj=['trajectory/tests/data/protein_only.xtc'])
        self.tr.discretize('rama', states=['A', 'E', 'O'])
        self.tr.find_keys()
    def test_init(self):
        self.msm = msm.SuperMSM([self.tr])
        assert self.msm is not None
        assert hasattr(self.msm, 'data')
        assert self.msm.data == [self.tr]
        assert self.msm.dt == 1.0
        print(self.msm.data)
    def test_merge_trajs(self):
    #   create fake trajectory to merge
        traj2 = traj.TimeSeries(distraj=['L','L','L','A'], dt = 2.0)
        traj2.keys = ['L','A']
        self.merge_msm = msm.SuperMSM([self.tr, traj2])
        assert sorted(self.merge_msm.keys) == ['A','E','L']
        assert isinstance(self.merge_msm.data[1] , traj.TimeSeries)
        for tr in self.merge_msm.data:
            print(tr.dt)
        assert self.merge_msm.dt == 1.0
if __name__ == "__main__":
    unittest.main()
