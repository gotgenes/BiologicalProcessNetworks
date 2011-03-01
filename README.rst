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

- Python v2.6 or v2.7 (Python 3 support not currently offered)
- ConflictsOptionParser_
- ConvUtils_
- fisher_
- NetworkX_ (v.1.0 or greater)
- SciPy_ (which depends on NumPy_)


The recommended way to install BPN is through the Python package
installer pip_, however, we also describe manual installation below.


Installation by pip
===================

pip will download and install BPN, as well as any Python package
dependencies that are not yet installed on your system or which require
upgrading. (**NOTE:** There is currently an exception for NumPy; see
[3]_.)


Local installation for non-privileged users
-------------------------------------------

If you do not have administrative, or do not wish to make a system-wide
installation of BPN, you can still install BPN and all its dependencies
using either a virtual enviroment or the user site-packages
installation.


Virtual environment installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It highly recommended to install BPN within its own virtual Python
environment using pip with `virtualenv`_. With the aid of
`virtualenvwrapper`_, the following steps will lead to a clean
installation of BPN:

::

  mkvirtualenv --distribute bpn
  pip install -E bpn numpy  # if numpy not installed
  pip install -E bpn BiologicalProcessNetworks

Again, assuming installation of virtualenvwrapper, to use the BPN
programs, do

::

  workon bpn
  bpln --help   # or appropriate command


User site-packages installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users who do not desire installation into a virtual environment may
still install locally into the user site-packages diretory using

::

  pip install --user BiologicalProcessNetworks


System-wide installation for users with administrative access
-------------------------------------------------------------

If you have administrative (e.g., sudo) access on your system, you may
install BPN system-wide with

::

  sudo pip install BiologicalProcessNetworks


Manual Installation
===================

Once you have installed the dependencies and have obtained and unpacked
the source for BPN (e.g., by using ``tar``), move into the top level
directory of the unpacked source and run

::

  python setup.py install


If you do not have administrative permissions for your computer, you can
install into the user-specific site-packages location with

::

  python setup.py install --user


.. [1] `Dotan-Cohen, D., *et al.* "Biological Process Linkage Networks."
   PLoS One. 2009. <http://dx.doi.org/10.1371/journal.pone.0005313>`_
.. [2] `Lasher, C., *et al.* "Discovering Networks of Perturbed
   Biological Processes in Hepatocyte Cultures." PLoS One. 2010.
   <http://dx.doi.org/10.1371/journal.pone.0015247>`_
.. [3] There is currently a bug in which pip attempts to install SciPy_
   before NumPy_. If your install fails during the installation of
   SciPy, try running ``pip install numpy`` (or local-install
   equivalent) prior to installing BPN.

.. _PyPI:
.. _Python Package Index: http://pypi.python.org/
.. _pip: http://pypi.python.org/pypi/pip
.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _virtualenvwrapper: http://www.doughellmann.com/projects/virtualenvwrapper/
.. _ConflictsOptionParser: http://pypi.python.org/pypi/ConflictsOptionParser/
.. _ConvUtils: http://pypi.python.org/pypi/ConvUtils/
.. _fisher: http://pypi.python.org/pypi/fisher/
.. _NetworkX: http://networkx.lanl.gov/
.. _NumPy: http://numpy.scipy.org/
.. _SciPy: http://scipy.org/
