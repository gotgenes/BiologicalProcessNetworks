==================================
Python Biological Process Networks
==================================

Python Biological Process Networks (PyBPN) provides programs to detect
connections between biological processes (called "links") based on gene
interaction, expression, and annotation data. A collection of
significant links and the participating processes forms a biological
process network, or BPN.

PyBPN provides three related programs for finding BPNs, each with
different objectives:

bpln
  Determines if processes are generally connected; an implementation of
  the algorithm described by Dotan-Cohen *et al.* [1]_.

cbpn
  Determines whether, under a particular comparison of conditions,
  connections between processes are perturbed; an implementation of the
  algorithm described by Lasher *et al.* [2]_.

mcmcbpn
  Similar to ``cbpn``, but attempts to discover the smallest set of
  connections which describes as much of the perturbation of interacting
  genes as possible.


############
Availability
############

PyBPN releases are available from the `Python Package Index`_ (`PyPI`_)
at http://pypi.python.org/pypi/BiologicalProcessNetworks

PyBPN's source code is hosted on `GitHub`_ at
https://github.com/gotgenes/BiologicalProcessNetworks


############
Installation
############

The recommended way to install PyBPN is through the Python package
installer pip_, as it helps automagically manage dependencies, however,
this document also provides instructions for manual installation.

PyBPN has several third-party `dependencies`_, described below.


.. _dependencies:

Dependencies
============

PyBPN depends on the following Python versions and external Python
Packages (all available from `PyPI`_):

- Python **2.6** or **2.7**. Python 3 is not currently supported;
  Python 2.5 and lower are unsupported. Check your Python version with
  ``python --version``. Obtain newer releases of Python from
  http://python.org/download/
- ConflictsOptionParser_
- ConvUtils_
- fisher_
- NetworkX_ (v.1.0 or greater)
- SciPy_ (which depends on NumPy_)


If you are installing PyBPN via pip, you only need to ensure that you have
an appropriate version of Python installed on your system. If you are
manually installing PyBPN, you will need to obtain and install all
dependencies through your own means (e.g., via ``apt``, ``yum``, ``.dmg``
installs, or from source, following the package's instructions).


Installation by pip
===================

pip_ will download and install PyBPN, as well as any Python package
dependencies that are not yet installed on your system or which require
upgrading.


System-wide installation for users with administrative access
-------------------------------------------------------------

If you have administrative (e.g., sudo) access on your system, you may
install PyBPN system-wide with

::

  sudo pip install BiologicalProcessNetworks

If you have not installed NumPy before hand, you may encounter an error
[3]_. In this case, try

::

  pip install numpy
  pip install BiologicalProcessNetworks


Local installation for non-privileged users
-------------------------------------------

If you do not have administrative, or do not wish to make a system-wide
installation of PyBPN, you can still install PyBPN and all its dependencies
using the user site-packages installation.

::

  pip install --user BiologicalProcessNetworks

If you have not installed NumPy before hand, you may encounter an error
[3]_. In this case, try

::

  pip install --user numpy
  pip install --user BiologicalProcessNetworks


Manual Installation
===================

Once you have installed all dependencies_ and have obtained and unpacked
the source for PyBPN (e.g., by using ``tar``), move into the top level
directory of the unpacked source and run

::

  python setup.py install


If you do not have administrative permissions for your computer, you can
install into the user-specific site-packages location with

::

  python setup.py install --user


#####
Usage
#####

All programs accept the ``-h``/``--help`` option. Provide this option to
get a full usage string from the program, including all available
options. Below is a summary of the usage for each program and details of
common options.


BPLN
====

TODO


CBPLN
=====

TODO


MCMCBPN
=======

``mcmcbpn`` calculates a BPN which explains as much gene expression
perturbation an underlying gene-gene (or protein-protein) response
network as possible, using as few process-process links as possible.
``mcmcbpn`` performs `Markov chain Monte Carlo (MCMC)`_ in order to
effectively consider all possible links simultaneously and select an
optimal subset of them.


Basic Usage
-----------

The basic usage of ``mcmcbpn`` is as follows::

  mcmcbpn [OPTIONS] INTERACTIONS_FILE ANNOTATIONS_FILE EXPRESSION_FILE

Each of the files is described below:

- ``INTERACTIONS_FILE``: a CSV file containing interactions. The file
  should have two columns with headings "interactor1" and
  "interactor2". It may have an optional column with the heading
  "weight", whose values will be used as the weight or confidence
  of the interaction. The file may have additional columns, which
  will be ignored.

- ``ANNOTATIONS_FILE``: a file containing annotations. The annotations
  file may be in one of two formats:

  - GMT format: if the file ends with the extension ".gmt", it is
    automatically parsed as a GMT-format file. The file is a
    tab-separated (TSV) format with no headers. The first column
    contains the annotation term. The second column contains a
    description. All following columns contain gene IDs for genes
    annotated by that term. `Full GMT format specification`_ is
    available from the `MSigDB and GSEA website`_.

  - Two-column format: The file should have a column titled
    "gene_id" which has the gene/gene product ID, and a column
    titled "term" which contains the term with which the
    gene/product is annotated. The file may have additional
    columns, which will be ignored.

- ``EXPRESSION_FILE``: a CSV file of gene (or gene product) expression
  values. The file should have a column titled "id" which has the
  gene (or gene product) ID, and a column titled "expression"
  which gives a value for the expression level, or difference in
  expression levels.

``mcmcbpn`` has a large number of options which can change its behavior,
either in terms of the algorithm and parameters used, or in terms of its
output. To get a full list of options, run ::

  mcmcbpn --help

Below are the most important options.


Algorithm and Parameter Options
-------------------------------

These are options which affect the algorithmic behavior or starting
state of ``mcmcbpn``.

- ``--burn-in=BURN_IN``: the number of steps to take before recording
  states in the Markov chain [default: ``1000000``]

- ``--steps=STEPS``: the number of steps through the Markov chain to
  observe [default: ``10000000``]

- ``--activity-threshold=ACTIVITY_THRESHOLD``: set the (differential)
  expression threshold at which a gene is considered active [default:
  ``-log10(0.05)``]

- ``--transition-ratio=TRANSITION_RATIO``: The target ratio of proposed
  link transitions to proposed parameter transitions [default: ``0.9``]

- ``--fixed-distributions``: use fixed distributions for link (and term)
  prior [implies ``--free-parameters]`` (**highly recommended**)

- ``--free-parameters``: parameters will be adjusted randomly, rather
  than incrementally (**recommended**)

- ``--disable-swaps``: disables swapping links as an option for
  transitions (**highly recommended**; will become the default option in
  future releases)


Output Options
--------------

These are options which affect the output file paths and file formats
for ``mcmcbpn``.

- ``--links-outfile=LINKS_OUTFILE``: the file to which the links results
  should be written [default: ``links_results.tsv``]

- ``--parameters-outfile=PARAMETERS_OUTFILE``: the file to which the
  parameters results should be written [default: parameter_results.tsv]

- ``--terms-outfile=TERMS_OUTFILE``: the file to which the terms results
  should be written [default: terms_results.tsv]

- ``--transitions-outfile=TRANSITIONS_OUTFILE``: the file to which the
  transitions data should be written [default: transitions.tsv]

- ``--detailed-transitions``: transitions file includes full information
  about each step's state (see also ``--bzip2`` below, as this can
  drastically increase the file size of the transitions outfile)

- ``--bzip2``: compress transitions file using bzip2 (**highly
  recommended**, the transitions file can consume a large amount of disk
  space, in proportion to the number of steps)

- ``--record-frequencies``: record the frequency of each state

- ``--frequencies-outfile=FREQUENCIES_OUTFILE``: the file to which
  frequency information should be written [default:
  ``state_frequencies.tsv``]

- ``--logfile=LOGFILE``: the file to which information for the run will
  be logged [default: ``mcmcbpn-TIMESTAMP.log``]


Output
------

The two principal files output by ``mcmcbpn`` are the links outfile and
the parameters outfile.

Links File
  This TSV file contains three columns: ``term1``, ``term2``, and
  ``probability``. ``term1`` and ``term2`` represent the two biological
  processes of a given link, and ``probability`` represents the
  probability that link should exist in the final biological process
  network (BPN) as determined by a given run of ``mcmcbpn``.

Parameters File
  This TSV file contains three columns: the first column, ``parameter``,
  represents the name of the given parameter. Names include the
  following:

  - ``link_false_neg``: proportion of interactions not explained by the
    BPN that should be
  - ``link_false_pos``: propotion of interactions explained by the BPN
    that should not be
  - ``link_prior``: the prior probability a link would be included in
    the BPN at all

  The second column, ``value``, shows a particular value for a given
  parameter. The third column, ``probability``, gives the estimated
  probability that the given ``parameter`` should assume the respective
  ``value`` in order to maximize the likelihood of the BPN.


.. [1] `Dotan-Cohen, D., *et al.* "Biological Process Linkage Networks."
   PLoS One. 2009. <http://dx.doi.org/10.1371/journal.pone.0005313>`_
.. [2] `Lasher, C., *et al.* "Discovering Networks of Perturbed
   Biological Processes in Hepatocyte Cultures." PLoS One. 2010.
   <http://dx.doi.org/10.1371/journal.pone.0015247>`_
.. [3] If your install fails during the installation of SciPy, try
   running ``pip install numpy`` (or local-install equivalent) prior to
   installing PyBPN.

.. _PyPI:
.. _Python Package Index: http://pypi.python.org/
.. _GitHub: https://github.com/
.. _pip: http://pypi.python.org/pypi/pip
.. _virtualenv: http://pypi.python.org/pypi/virtualenv
.. _virtualenvwrapper: http://www.doughellmann.com/projects/virtualenvwrapper/
.. _ConflictsOptionParser: http://pypi.python.org/pypi/ConflictsOptionParser/
.. _ConvUtils: http://pypi.python.org/pypi/ConvUtils/
.. _fisher: http://pypi.python.org/pypi/fisher/
.. _NetworkX: http://networkx.lanl.gov/
.. _NumPy: http://numpy.scipy.org/
.. _SciPy: http://scipy.org/
.. _`Markov chain Monte Carlo (MCMC)`: http://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo
.. _`Full GMT format specification`: http://www.broadinstitute.org/cancer/software/gsea/wiki/index.php/Data_formats#GMT:_Gene_Matrix_Transposed_file_format_.28.2A.gmt.29
.. _MSigDB and GSEA website: http://www.broadinstitute.org/gsea/
