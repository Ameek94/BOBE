Installation Guide
==================

Requirements
------------

JaxBo requires Python 3.10 or later and has the following dependencies:

Core Dependencies
~~~~~~~~~~~~~~~~~

- **JAX**: For automatic differentiation and acceleration
- **JAXlib**: JAX's XLA-based backend
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing
- **NumPyro**: Probabilistic programming
- **Optax**: Gradient-based optimization

Cosmology Dependencies
~~~~~~~~~~~~~~~~~~~~~~

- **Cobaya**: Cosmological parameter estimation
- **GetDist**: Analysis and plotting of Monte Carlo samples

Visualization Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Matplotlib**: Plotting
- **tqdm**: Progress bars

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Clone the repository:

.. code-block:: bash

   git clone https://github.com/Ameek94/JaxBo.git
   cd JaxBo

2. Install the package:

.. code-block:: bash

   python -m pip install .

For development installation (allows editing the source code):

.. code-block:: bash

   python -m pip install -e .

With Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~

Install with development tools:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Install with documentation tools:

.. code-block:: bash

   python -m pip install -e ".[docs]"

Virtual Environment Setup
--------------------------

It's recommended to use a virtual environment to avoid dependency conflicts:

Using conda
~~~~~~~~~~~

.. code-block:: bash

   conda create -n jaxbo python=3.10
   conda activate jaxbo
   python -m pip install -e .

Using venv
~~~~~~~~~~

.. code-block:: bash

   python -m venv jaxbo_env
   source jaxbo_env/bin/activate  # On Windows: jaxbo_env\Scripts\activate
   python -m pip install -e .

GPU Support
-----------

For GPU acceleration, you'll need to install the appropriate JAXlib version:

CUDA Support
~~~~~~~~~~~~

.. code-block:: bash

   # Install CUDA-compatible JAX (check JAX documentation for latest versions)
   pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

TPU Support
~~~~~~~~~~~

.. code-block:: bash

   # For TPU support (Google Cloud Platform)
   pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

Verification
------------

To verify your installation, run:

.. code-block:: python

   import jaxbo
   print(f"JaxBo version: {jaxbo.__version__}")
   
   # Check JAX backend
   import jax
   print(f"JAX backend: {jax.default_backend()}")
   print(f"JAX devices: {jax.device_count()}")

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**ImportError: No module named 'jax'**
   Ensure JAX is properly installed. Try reinstalling with:
   
   .. code-block:: bash
   
      pip install --upgrade jax jaxlib

**CUDA/GPU not detected**
   Make sure you have the correct CUDA-compatible JAXlib version installed.
   Check your CUDA version with ``nvidia-smi`` and install the corresponding JAXlib.

**Memory issues**
   If you encounter out-of-memory errors:
   
   - Reduce the ``max_gp_size`` parameter in BOBE
   - Enable JAX memory preallocation: ``export XLA_PYTHON_CLIENT_PREALLOCATE=false``

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~~~

**macOS with Apple Silicon**
   JAX has native support for Apple Silicon. No special installation required.

**Windows**
   GPU support on Windows requires WSL2 for CUDA functionality.

Getting Help
------------

If you encounter installation issues:

1. Check the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
2. Review the `GitHub Issues <https://github.com/Ameek94/JaxBo/issues>`_
3. Contact the developers (see main documentation page)
