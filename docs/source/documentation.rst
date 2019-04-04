.. _documentation:

Documentation
=============
MasterMSM is a Python package that is divided in three main subpackages. 
This way of structuring the code derives from the three main types of 
objects that are constructed. First, there are trajectories, which 
result in objects of the ``TimeSeries`` class; second, there are dynamical
models, which come in the form of instances of the ``MSM`` class; finally,
dynamical models can be postprocessed into simple, few-state models, which
we generate as ``PCCA`` class objects.

Trajectory module
-----------------
This module contains everything necessary to get your time series data
into MasterMSM. The main class object within this module is the TimeSeries
object.

.. currentmodule:: mastermsm

.. autosummary::
    :toctree: 

    trajectory 


MSM module
----------
.. currentmodule:: mastermsm

.. autosummary::
    :toctree: 

    msm 


PCCA module
-----------
.. currentmodule:: mastermsm

.. autosummary::
    :toctree: 

    pcca
    
Examples
--------
We have put together a few simple Python notebooks to help you learn the basics
of the MasterMSM package. They are based on data derived from either model systems
or from molecular dynamics simulations of some simple (albeit realistic) biomolecules.
You can find the notebooks in the following 
`link <https://github.com/daviddesancho/MasterMSM/tree/develop/examples>`_.

