Installation
============
You can install MasterMSM by simply downloading the package from the 
`GitHub repository <https://github.com/daviddesancho/MasterMSM>`_
and using the standard installation instructions for packages built
using `Distutils <https://docs.python.org/3/distutils/index.html>`_.

.. code-block:: bash

   git clone http://github.com/daviddesancho/mastermsm destination/mastermsm
   cd destination/mastermsm
   python setup.py install --user

Parallel processing in Python and MasterMSM
-------------------------------------------
In MasterMSM we make ample use of the ``multiprocessing`` library, which
for MacOS X can conflict with non-Python libraries. In the past we have
found this to be a problem that can result in segmentation faults. 
Digging in the internet I found a workaround for this problem, by setting 
the following environment variable

.. code-block:: bash

   export VECLIB_MAXIMUM_THREADS=1

This should be set in the terminal before you start your Python session
in case you meet this problem.

