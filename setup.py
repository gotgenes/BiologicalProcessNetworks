#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from setuptools import setup

import os.path

scripts = ['bpln', 'cbpn', 'mcmcbpn', 'sabpn']
scripts = [os.path.sep.join(('scripts', script)) for script in scripts]

setup(
    name='BiologicalProcessNetworks',
    version='1.0a4',
    author='Christopher D. Lasher',
    author_email='chris.lasher@gmail.com',
    install_requires=[
        'bitarray',
        'ConflictsOptionParser',
        'ConvUtils<2.0',
        'fisher',
        'networkx>=1.0',
        'numpy',
        'scipy'
    ],
    packages=['bpn', 'bpn.mcmc', 'bpn.tests'],
    scripts=scripts,
    url='http://pypi.python.org/pypi/BiologicalProcessNetworks',
    license='MIT License',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
    description=("Identify significant connections between "
            "biological processes using gene interaction networks."),
    long_description=open('README.rst').read(),
)
