import unittest
import mdtraj as md
from mastermsm.trajectory import traj

class TestMDtraj(unittest.TestCase):
    def setUp(self):
        self.traj = md.load('trajectory/tests/data/protein_only.xtc', \
                top='trajectory/tests/data/alaTB.gro')

    def test_traj(self):
        assert self.traj.n_atoms == 19
        assert self.traj.timestep == 1.0

class UseMDtraj(unittest.TestCase):
    def setUp(self):
        self.tr = traj.TimeSeries(top='trajectory/tests/data/alaTB.gro', \
                traj=['trajectory/tests/data/protein_only.xtc'])
        pass

#    def test_atributes(self):
#        self.tr
#    def test_discretize(self):
#        assert self.tr.n_traj == 1

if __name__ == "__main__":
    unittest.main()
