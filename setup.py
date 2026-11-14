#!/usr/bin/env python

# Setup script for the MasterMSM package

import os
import re
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def version():
    init = read(os.path.join('mastermsm', '__init__.py'))
    return re.search(r'__version__ = "(.+)"', init).group(1)

setup(
		name='MasterMSM',
		version=version(),
		description='Algorithms to construct master equation / Markov state models',
		url='https://github.com/BioKT/MasterMSM',
		author='David De Sancho',
		author_email='daviddesancho.at.gmail.com',
		license='GPL-2.0',
        	packages=find_packages(),
		keywords= "markov state model",
		long_description=read('README.md'),
		long_description_content_type='text/markdown',
		python_requires='>=3.10',
		install_requires=[
			'numpy',
			'scipy',
			'matplotlib',
			'networkx',
			'mdtraj',
			'scikit-learn',
		],
		extras_require={
			# Discretizer.hdbscan uses hdbscan.approximate_predict,
			# which sklearn.cluster.HDBSCAN does not provide
			'hdbscan': ['hdbscan'],
		},
		classifiers = ["""\
				Development Status :: 3 - Alpha
				License :: OSI Approved :: GNU General Public License v2 (GPLv2)
				Operating System :: POSIX :: Linux
				Operating System :: MacOS
				Programming Language :: Python :: 3.10
				Programming Language :: Python :: 3.11
				Programming Language :: Python :: 3.12
				Topic :: Scientific/Engineering :: Bio-Informatics
				Topic :: Scientific/Engineering :: Chemistry
				"""]
		)
