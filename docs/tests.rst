*****************
Tests and checks
*****************

Functions for checking that the inference model is working correctly. Run within :py:meth:`get_trace() <stratmc.inference>` in :py:mod:`stratmc.inference` after sampling is complete.

.. currentmodule:: stratmc.tests

.. todo::
    Write a master function that can be run after every inference (ideally inside of get_trace) to check that there are no issues

.. autosummary::
  :nosignatures:
  
  check_superposition
  check_section_superposition
  check_detrital_ages
  check_intrusive_ages
  
.. automodule:: stratmc.tests
  :members:
  