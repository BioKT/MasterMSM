Installation
============
MasterMSM requires Python >= 3.10. Install it from
`PyPI <https://pypi.org/project/MasterMSM/>`_ with ``pip``:

.. code-block:: bash

   pip install MasterMSM

To work on the code, install it from the
`GitHub repository <https://github.com/BioKT/MasterMSM>`_ instead:

.. code-block:: bash

   git clone https://github.com/BioKT/MasterMSM
   cd MasterMSM
   pip install -e .

Dependencies
------------
``pip`` installs the required dependencies automatically: numpy, scipy,
matplotlib, networkx, `MDTraj <https://mdtraj.org>`_ and scikit-learn.
With conda, you may prefer to install them from conda-forge first:

.. code-block:: bash

   conda install -c conda-forge numpy scipy matplotlib \
       networkx mdtraj scikit-learn

Clustering with HDBSCAN additionally requires the optional
`hdbscan <https://github.com/scikit-learn-contrib/hdbscan>`_ package,
which can be installed as an extra:

.. code-block:: bash

   pip install "MasterMSM[hdbscan]"

or on its own with ``pip install hdbscan``.

Testing the installation
------------------------
The test suite uses ``pytest`` and downloads a small amount of MD data
on its first run:

.. code-block:: bash

   pip install pytest
   pytest --pyargs mastermsm

Parallel processing in Python and MasterMSM
-------------------------------------------
MasterMSM uses the ``multiprocessing`` library to calculate count
matrices and bootstrap errors. When running MasterMSM from a Python
script (not from a Jupyter notebook), protect the main code with

.. code-block:: python

   if __name__ == "__main__":
       ...

since on macOS and Windows worker processes re-import the script.

On macOS, ``multiprocessing`` can also conflict with the Accelerate
linear algebra libraries and cause segmentation faults. If you meet
this problem, set the following environment variable in the terminal
before starting your Python session:

.. code-block:: bash

   export VECLIB_MAXIMUM_THREADS=1
