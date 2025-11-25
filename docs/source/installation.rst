Installation Guide
==================

Requirements
------------

JaxBO requires Python 3.11 or 3.12 and has the following core dependencies:

Core Dependencies
~~~~~~~~~~~~~~~~~

- **JAX**: For automatic differentiation and jit compilation (and in the future GPU/TPU acceleration...)
- **JAXlib**: JAX's XLA-based backend
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing and optimization
- **NumPyro**: Probabilistic programming library
- **Dynesty**: Nested sampling
- **scikit-learn**: SVM classifiers
- **TensorFlow Probability**: Statistical distributions

Additional Dependencies
~~~~~~~~~~~~~~~~~~~~~~~

- **GetDist**: Analysis and plotting of Monte Carlo samples
- **Matplotlib**: Visualization
- **tqdm**: Progress bars

.. Optional Dependencies
.. ~~~~~~~~~~~~~~~~~~~~~

.. JaxBO has several optional dependencies for extended functionality:

.. **Neural Network Classifiers (Flax + Optax)**
..   Install with: ``pip install 'jaxbo[nn]'``
  
..   - Enables MLPClassifier and EllipsoidClassifier for GP filtering
..   - Required for using ``clf_type='nn'`` or ``clf_type='ellipsoid'``
  
.. **Cobaya Likelihoods**
..   Install with: ``pip install 'jaxbo[cobaya]'``
  
..   - Enables CobayaLikelihood class for cosmological likelihoods
..   - Required for interfacing with Cobaya cosmological models
  
.. **MPI Parallelization**
..   Install with: ``pip install 'jaxbo[mpi]'``
  
..   - Enables parallel likelihood evaluation and gp fitting across multiple processes
..   - Highly recommended for expensive likelihoods (evaluation time > 1 second)

.. **All Optional Dependencies**
..   Install with: ``pip install 'jaxbo[all]'``

.. .. note::
..    By default, only SVM classifiers are available. The core BO functionality 
..    works with scipy optimization and does not require optax.

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

3. (Optional) Install with additional features:

.. code-block:: bash

   # For neural network classifiers, by default svm is available through scikit-learn
   pip install '.[nn]'
   
   # In case you want to use cosmological likelihoods through the Cobaya interface
   pip install '.[cobaya]'
   
   # For MPI support
   pip install '.[mpi]'
   
   # For everything
   pip install '.[all]'

For development installation (allows editing the source code):

.. code-block:: bash

   python -m pip install -e .

Using Conda (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you face installation issues related to incompatible versions of packages, you can set up 
an environment with the exact package versions JaxBO was developed and tested with:

.. code-block:: bash

   # Create environment from the minimal essential packages
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate jaxbo
   
   # Install JaxBO in development mode
   pip install -e .

Virtual Environment Setup
--------------------------

It's recommended to use a virtual environment to avoid dependency conflicts:

Using conda
~~~~~~~~~~~

.. code-block:: bash

   conda create -n jaxbo python=3.12
   conda activate jaxbo
   python -m pip install .

Using venv
~~~~~~~~~~

.. code-block:: bash

   python -m venv jaxbo_env
   source jaxbo_env/bin/activate  # On Windows: jaxbo_env\Scripts\activate
   python -m pip install .

GPU/TPU Support
-----------
In progress...

.. For GPU acceleration, you'll need to install the appropriate JAXlib version:

.. CUDA Support
.. ~~~~~~~~~~~~

.. .. code-block:: bash

..    # Install CUDA-compatible JAX (check JAX documentation for latest versions)
..    pip install --upgrade "jax[cuda12]"

.. .. note::
..    JAX GPU support requires NVIDIA GPU with CUDA installed. See the 
..    `JAX documentation <https://jax.readthedocs.io/en/latest/installation.html>`_ 
..    for detailed GPU installation instructions.

.. TPU Support
.. ~~~~~~~~~~~

.. .. code-block:: bash

..    # For TPU support (Google Cloud Platform)
..    pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

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

.. Troubleshooting
.. ---------------

.. Common Issues
.. ~~~~~~~~~~~~~

.. **ImportError: No module named 'jax'**
..    Ensure JAX is properly installed. Try reinstalling with:
   
..    .. code-block:: bash
   
..       pip install --upgrade jax jaxlib

.. **CUDA/GPU not detected**
..    Make sure you have the correct CUDA-compatible JAXlib version installed.
..    Check your CUDA version with ``nvidia-smi`` and install the corresponding JAXlib.

.. **Memory issues**
..    If you encounter out-of-memory errors:
   
..    - Reduce the ``max_gp_size`` parameter in BOBE
..    - Enable JAX memory preallocation: ``export XLA_PYTHON_CLIENT_PREALLOCATE=false``

.. Platform-Specific Notes
.. ~~~~~~~~~~~~~~~~~~~~~~~~

.. **macOS with Apple Silicon**
..    JAX has native support for Apple Silicon. No special installation required.

.. **Windows**
..    GPU support on Windows requires WSL2 for CUDA functionality.

Getting Help
------------

If you encounter installation issues:

1. Check the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_
2. Review the `GitHub Issues <https://github.com/Ameek94/JaxBo/issues>`_
3. Contact the developers (see main documentation page)
