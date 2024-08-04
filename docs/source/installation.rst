*****************
Installation
*****************

If you're new to Python, we recommend using `Anaconda <https://www.anaconda.com/>`_ to install Python on your machine. You can then manage packages from the terminal using ``conda``.

To create a new conda environment for StratMC, run:

.. code-block:: bash

  conda create --name stratmc_env

Before installing StratMC, activate the new environment and install ``pip``:

.. code-block:: bash

  conda activate stratmc_env
  conda install pip

You can then install StratMC and its dependencies using ``pip`` (note that the ``--pre`` flag is required to install the current version, which is a pre-release), or by compiling directly from the GitHub repository:

PIP
#####

.. code-block:: bash

  pip install stratmc --pre

Latest (*unstable*)
####################

.. code-block:: bash

  pip install git+https://github.com/sedmonsond/stratmc


Installing on Apple Silicon
############################
On Apple Silicon machines (M1 chip or later), sampling is significantly faster when the Apple Accelerate BLAS library is used, rather than the default OpenBLAS library. After installing StratMC in a new conda environment, run:

.. code-block:: bash

  conda install -c conda-forge "libblas=*=*accelerate"

If you're managing your packages with pip in a virtual environment, instead of with conda, activate the environment and run:

.. code-block:: bash

  pip install cython pybind11
  pip install --no-binary :all: --no-use-pep517 numpy
