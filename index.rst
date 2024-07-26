.. title:: StratMC Documentation

.. figure:: stratmc.png
   :width: 60 %
   :alt: StratMC logo
   :align: center

===========================================================
Reconstructing proxy signals from the stratigraphic record
===========================================================

StratMC is a statistical framework for reconstructing past Earth system change using sediment-hosted proxy data. It is built on the Python probabilistic programming library `PyMC <https://www.pymc.io/welcome.html>`_, which provides a flexible toolbox for constructing Bayesian models and sampling their posteriors using Markov chain Monte Carlo (MCMC) methods. 

Using geochemical proxy observations and geological age constraints from multiple stratigraphic sections, StratMC simultaneously infers the global proxy signal recorded by all sections and builds an age model for each section. For a complete description of the model, see Edmonsond & Dyer (submitted to *Geoscientific Model Development*). Key aspects of the model structure are summarized on the :doc:`model description <docs/model_description>` page. 

The StratMC Python package can be :doc:`installed <docs/installation>` from `PyPI <https://pypi.org>`_ using the `pip package installer <https://packaging.python.org/en/latest/guides/tool-recommendations/>`_ . The :doc:`API reference <docs/api>` catalogs built-in functions for processing data, running the inference model, and plotting the results. For a full list of resources, visit the :doc:`User Guide <docs/userguide>`.  

Getting Started
----------------
* :doc:`Installation <docs/installation>`
* :doc:`Quick start guide <docs/quickstart>`
* :doc:`Model description <docs/model_description>`
* :doc:`Example notebooks <docs/examples>`
* :doc:`API reference <docs/api>`
* :doc:`Data table formatting <docs/datatable>`


.. toctree::
    :maxdepth: 1
    :hidden:

    User Guide <docs/userguide>
    API Reference<docs/api>
    Examples<docs/examples>
    Installation<docs/installation>