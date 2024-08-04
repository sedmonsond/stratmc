
.. image:: https://raw.githubusercontent.com/sedmonsond/stratmc/main/docs/source/logos/stratmc.svg
    :height: 100px
    :alt: StratMC logo
    :align: center

|Build Status| |Coverage|

.. |Build Status| image:: https://github.com/sedmonsond/stratmc/workflows/pytest/badge.svg
   :target: https://github.com/sedmonsond/stratmc/actions
.. |Coverage| image:: https://codecov.io/gh/sedmonsond/stratmc/graph/badge.svg?token=P0ANAUP3BX
 :target: https://codecov.io/gh/sedmonsond/stratmc


StratMC is a statistical framework for reconstructing past Earth system change using sediment-hosted proxy data. It is built on the Python probabilistic programming library `PyMC <https://www.pymc.io/welcome.html>`_, which provides a flexible toolbox for constructing Bayesian models and sampling their posteriors using Markov chain Monte Carlo (MCMC) methods.

Using geochemical proxy observations and geological age constraints from multiple stratigraphic sections, StratMC simultaneously infers the global proxy signal recorded by all sections and builds an age model for each section. For a complete description of the model, see Edmonsond & Dyer (submitted to *Geoscientific Model Development*).

To get started with StratMC, check out the `package documentation <https://stratmc.readthedocs.io/>`_.
