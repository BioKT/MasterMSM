import unittest
import mdtraj as md
from mastermsm.trajectory import traj_lib
from mastermsm.trajectory import traj

class TestMDTrajLib(unittest.TestCase):
    def test_inrange(self):
        assert traj_lib._inrange(2, 1, 3) == 1
        assert traj_lib._inrange(0, 1, 2) == 0
        assert traj_lib._inrange(1, 1, 2) == 0
    def test_inbounds(self):
        assert traj_lib._inbounds({}, 90, 90)
    def test_stategrid(self):
        assert traj_lib._stategrid(10, 10, 20)
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
