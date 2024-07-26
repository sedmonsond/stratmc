.. toctree::
  :hidden:
  :maxdepth: 2

.. _datatable_target: 

******************
Data table format
******************

The inference model requires two inputs: proxy data for multiple stratigraphic sections and age constraints (at least a minimum and maximum age for each section). Proxy data and age constraints should be saved in separate ``.csv`` files formatted according to the tables below. 

Stratigraphic proxy data
-------------------------

The table below describes the values expected in the data table (``.csv`` file) with proxy data for each section. Each entry corresponds to a single sample. Optional parameters that you do not wish to specify should be left blank, and will be replaced with the default value when the data are loaded with :py:meth:`load_data() <stratmc.data>` in :py:mod:`stratmc.data`. 

.. csv-table:: 
   :file: datatable.csv
   :header: "Parameter", "Description"
   :widths: 20, 80
   
Age constraints
---------------

The table below describes the values expected in the data table (``.csv`` file) with age constraints for each section. Each entry corresponds to a single age constraint from one of the sections included in the ``Stratigraphic proxy data`` table. Optional parameters that you do not wish to specify should be left blank, and will be replaced with the default value when the data are loaded with :py:meth:`load_data() <stratmc.data>` in :py:mod:`stratmc.data`.
   
.. csv-table:: 
   :file: age_datatable.csv
   :header: "Parameter", "Description"
   :widths: 20, 80

