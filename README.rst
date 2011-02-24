===========================
Biological Process Networks
===========================

Biological Process Networks (BPN) provides programs to detect
connections between biological processes based on gene interaction,
expression, and annotation data.

BPN provides three related programs:

bpln
  Determines if processes are generally connected; an implementation of
  the algorithm described by Dikla Dotan-Cohen, *et al.* [1]_.

cbpn
  Determines whether, under a particular comparison of conditions,
  connections between processes are perturbed; an implementation of the
  algorithm described by Christopher Lasher, *et al.* [2]_.

mcmcbpn
  Similar to cbpn, but attempts to discover the smallest set of
  connections which describes as much of the perturbation of interacting
  genes as possible.


------------
Installation
------------

BPN requires the following external Python Packages, all available from
the `Python Package Index`_:

- ConflictsOptionParser_
- ConvUtils_
- NetworkX_ (v.1.0 or greater)
- SciPy_

Installation by pip
===================

You may install BPN by running
::

  pip install BiologicalProcessNetworks


Manual Installation
===================

Once you have installed the dependencies and have obtained and unpacked
the source for BPN (e.g., by using ``tar``), move into the top level
directory of the unpacked source and run
::

  python setup.py install


If you do not have administrative permissions for your computer, you can
alternatively install into the user-specific site-packages location with
::

  python setup.py install --user


.. [1] `Dotan-Cohen, D., *et al.* "Biological Process Linkage Networks."
   PLoS One. 2009. <http://dx.doi.org/10.1371/journal.pone.0005313>`_
.. [2] `Lasher, C., *et al.* "Discovering Networks of Perturbed
   Biological Processes in Hepatocyte Cultures." PLoS One. 2010.
   <http://dx.doi.org/10.1371/journal.pone.0015247>`_

.. _PyPI:
.. _ConflictsOptionParser: http://pypi.python.org/pypi/ConflictsOptionParser/
.. _Python Package Index: http://pypi.python.org/
.. _ConvUtils: http://pypi.python.org/pypi/ConvUtils/
.. _NetworkX: http://networkx.lanl.gov/
.. _SciPy: http://scipy.org/
