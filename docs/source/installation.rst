Installation
============
MasterMSM requires Python >= 3.10. Install it from
`conda-forge <https://anaconda.org/conda-forge/mastermsm>`_ with conda:

.. code-block:: bash

   conda install -c conda-forge mastermsm

or from `PyPI <https://pypi.org/project/MasterMSM/>`_ with ``pip``:

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
Both conda and ``pip`` install the required dependencies
automatically: numpy, scipy, matplotlib, networkx,
`MDTraj <https://mdtraj.org>`_ and scikit-learn.

Clustering with HDBSCAN additionally requires the optional
`hdbscan <https://github.com/scikit-learn-contrib/hdbscan>`_ package:

.. code-block:: bash

   conda install -c conda-forge hdbscan

or, with ``pip``, as an extra: ``pip install "MasterMSM[hdbscan]"``.

Testing the installation
------------------------
The test suite uses ``pytest``:

.. code-block:: bash

   pip install pytest
   pytest --pyargs mastermsm

Tests that depend on external reference trajectories are skipped
automatically in standard user installations (or when
``MASTERMSM_NO_DOWNLOAD=1`` is set without cached data).

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
