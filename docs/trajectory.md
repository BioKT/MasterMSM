# Parsing trajectories: the `trajectory` module

All the aspects related with parsing trajectory files in MasterMSM are included
in the `trajectory` module. 

The first step when consists in reading discretized trajectory 
files. This is done by constructing instances of the `TimeSeries` class. The file
format for the time series is very simple. One needs something like

```
0.0 00000
0.2 00001
0.4 00001
...
```

where the first column is the time stamp for the snapshots in the simulation
and the second column is the list of states. The time stamps can be omitted, 
in which case integer units of time will be used. The states can be anything
that fits in a string, as they will be read as strings by MasterMSM. 

For each trajectory we will generate an instance of the `TimeSeries` class in the
following way:

```
from mastermsm import trajectory as traj

mytraj0 = traj.TimeSeries("filename0")
mytraj1 = traj.TimeSeries("filename1")
...
```

Each instance of the `TimeSeries` class has its own set of attributes: the 
value for `dt`, the `keys` or names of the states, the `filename` that 
the data was read from, and the time series itself, formed by two lists
of time-stamps (`time`) and states (`states`).
