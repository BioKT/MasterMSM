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

MSM module
----------

PCCA module
-----------
