import unittest
import mdtraj as md
from mastermsm.trajectory import traj_lib
from mastermsm.trajectory import traj
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

    def test_traj(self):
        assert self.traj is not None
        assert self.traj.n_atoms == 19
        assert self.traj.timestep == 1.0
        assert self.traj.n_residues == 3
        assert self.traj.n_frames == 10003


class UseMDtraj(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                traj=['trajectory/tests/data/protein_only.xtc'])

    def test_atributes(self):
        assert self.tr.mdt is not None


#    def test_discretize(self):
#        assert self.tr.n_traj == 1

if __name__ == "__main__":
    unittest.main()
