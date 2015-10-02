#!/usr/bin/env python

# Setup script for bestmsm package

import os
from setuptools import setup,find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
		name='MasterMSM',
		version='0.1dev',
		description='Algorithms to construct master equation / Markov state models',
		url='http://github.com/daviddesancho/MasterMSM',
		author='David De Sancho',
		author_email='daviddesancho.at.gmail.com',
		license='GPL',
		packages=find_packages(),
		keywords= "markov state model",
		long_description=read('README.md'),
		classifiers = ["""\
				Development Status :: 1 - Planning
				Operating System :: POSIX :: Linux
				Operating System :: MacOS
				Programming Language :: Python :: 2.7
				Topic :: Scientific/Engineering :: Bio-Informatics
				Topic :: Scientific/Engineering :: Chemistry
				"""]
		)
