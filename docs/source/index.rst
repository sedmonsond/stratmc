.. title:: StratMC Documentation

.. figure:: logos/stratmc.svg
   :width: 60 %
   :alt: StratMC logo
   :align: center

============================================================
Reconstructing proxy signals from the stratigraphic record
============================================================

StratMC is a statistical framework for reconstructing past Earth system change using sediment-hosted proxy data. It is built on the Python probabilistic programming library `PyMC <https://www.pymc.io/welcome.html>`_, which provides a flexible toolbox for constructing Bayesian models and sampling their posteriors using Markov chain Monte Carlo (MCMC) methods.

Using geochemical proxy observations and geological age constraints from multiple stratigraphic sections, StratMC simultaneously infers the global proxy signal recorded by all sections and builds an age model for each section. For a complete description of the model, see Edmonsond & Dyer (submitted to *Geoscientific Model Development*).

The StratMC Python package can be :doc:`installed <installation>` from `PyPI <https://pypi.org>`_ using the `pip package installer <https://packaging.python.org/en/latest/guides/tool-recommendations/>`_. The :doc:`API reference <api>` catalogs built-in functions for processing data, running the inference model, and plotting the results. For a full list of resources, visit the :doc:`User Guide <userguide>`.

Getting Started
----------------
* :doc:`Installation <installation>`
* :doc:`Quick start guide <quickstart>`
* :doc:`Example notebooks <examples>`
* :doc:`API reference <api>`
* :doc:`Data table formatting <datatable>`


.. toctree::
    :maxdepth: 1
    :hidden:

    User Guide <userguide>
    API Reference<api>
    Examples<examples>
    Installation<installation>
