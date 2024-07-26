.. title:: StratMC Documentation

Package Overview
----------------

.. figure:: stratmc.png
   :width: 60 %
   :alt: StratMC logo
   :align: center

StratMC is a statistical framework for reconstructing past Earth system change using sediment-hosted proxy data. It is built on the Python probabilistic programming library `PyMC <https://www.pymc.io/welcome.html>`_, which provides a flexible toolbox for constructing Bayesian models and sampling their posteriors using Markov chain Monte Carlo (MCMC) methods. 

Using geochemical proxy observations and geological age constraints from multiple stratigraphic sections, StratMC simultaneously infers the global proxy signal recorded by all sections and builds an age model for each section. For a complete description of the model, see Edmonsond & Dyer (submitted to *Geoscientific Model Development*).

The StratMC Python package can be :doc:`installed <docs/installation>` from `PyPI <https://pypi.org>`_ using the `pip package installer <https://packaging.python.org/en/latest/guides/tool-recommendations/>`_ . The :doc:`API Reference <docs/api>` catalogs built-in functions for processing data, running the inference model, and plotting the results. For example notebooks, refer to the online `package documentation <https://sedmonsond.github.io/stratmc/>`_ (https://sedmonsond.github.io/stratmc/). 



.. toctree::
   :maxdepth: 2
   :hidden:

   Installation <docs/installation>
   Quick start guide <docs/quickstart>
   Data table format <docs/datatable>
   API Reference<docs/api>
