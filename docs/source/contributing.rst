Contributing to BOBE
====================

We welcome contributions to BOBE! This guide explains how to contribute to the project.

Development Setup
-----------------

1. Fork and Clone
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/BOBE.git
   cd BOBE

2. Install in Development Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create environment (optional)
   conda create -n BOBE-dev python=3.12
   conda activate BOBE-dev
   
   # Install in editable mode
   pip install -e .

Documentation
-------------

Build documentation locally:

.. code-block:: bash

   cd docs/
   make html

The built documentation will be in ``docs/build/html/``.

Submitting Changes
------------------

1. Create a feature branch:

   .. code-block:: bash

      git checkout -b feature/my-new-feature

2. Make your changes

3. Test your changes with examples

4. Commit your changes:

   .. code-block:: bash

      git commit -m "Add my new feature"

5. Push to your fork and submit a pull request to the main branch

Reporting Issues
----------------

Please report bugs and feature requests via the `GitHub issue tracker <https://github.com/Ameek94/BOBE/issues>`_.

When reporting bugs, please include:

- Your operating system and Python version
- BOBE version (or commit hash if using git)
- Steps to reproduce the issue
- Expected vs. actual behavior
- Minimal code example demonstrating the problem (if applicable)

Questions
---------

For questions about using BOBE, please:

- Check the documentation and examples first
- Search existing GitHub issues
- Open a new issue if your question hasn't been addressed

Thank you for contributing to BOBE!
