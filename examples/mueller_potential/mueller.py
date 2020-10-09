#!/bin/env python

#Copyright 2020 Robert T. McGibbon

#Permission is hereby granted, free of charge, to any person i
# obtaining a copy of this software and associated documentation 
# files (the "Software"), to deal in the Software without restriction, 
# including without limitation the rights to use, copy, modify, 
# merge, publish, distribute, sublicense, and/or sell copies of the 
# Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be 
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND 
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR 
# OTHER DEALINGS IN THE SOFTWARE.

from simtk.unit import kelvin, picosecond, femtosecond, nanometer, dalton
import simtk.openmm as mm
import matplotlib.pyplot as plt
import numpy as np

class MullerForce(mm.CustomExternalForce):
    """
    OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting.


    """
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-200, -100, -170, 15]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j], AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            expression += '''+ {AA}*exp({aa}*(x - {XX})^2 + {bb}*(x - {XX}) 
                               *(y - {YY}) + {cc}*(y - {YY})^2)'''.format(**fmt)
        super(MullerForce, self).__init__(expression)
    
    @classmethod
    def potential(cls, x, y):
        "Compute the potential at a given point x,y"
        value = 0
        for j in range(4):
            value += cls.AA[j]*np.exp(cls.aa[j]*(x - cls.XX[j])**2 + \
                cls.bb[j]*(x - cls.XX[j])*(y - cls.YY[j]) \
                + cls.cc[j]*(y - cls.YY[j])**2)
        return value

    @classmethod
    def plot(cls, ax=None, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 200.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = cls.potential(xx, yy)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        if ax is None:
            ax = plt
        ax.contourf(xx, yy, V.clip(max=200), 40, alpha=0.4, **kwargs)

if __name__ == "__main__":
    ##############################################################################
    # Global parameters
    ##############################################################################
    
    # each particle is totally independent, propagating under the same potential
    mass = 1.0*dalton
    temperature = 750*kelvin
    friction = 100/picosecond
    timestep = 10.0*femtosecond
    
    # Choose starting conformations uniform on the grid between (-1.5, -0.2) and (1.2, 2)
    startingPositions = (np.random.rand(1, 3)*np.array([2.7, 1.8, 1])) \
            + np.array([-1.5, -0.2, 0])
    
    system = mm.System()
    mullerforce = MullerForce()
    system.addParticle(mass)
    mullerforce.addParticle(0, [])
    system.addForce(mullerforce)
    
    integrator = mm.LangevinIntegrator(temperature, friction, timestep)
    context = mm.Context(system, integrator)
    context.setPositions(startingPositions)
    context.setVelocitiesToTemperature(temperature)
    
    traj = []
    for i in range(int(1e6)):
        traj.append(
                context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)[0])
        integrator.step(200)
    traj = np.vstack(traj)
    
    fig, ax = plt.subplots(figsize=(4,4))
    MullerForce.plot(ax=ax)
    ax.plot(traj[:,0], traj[:,1], c='k', lw=0.1)
