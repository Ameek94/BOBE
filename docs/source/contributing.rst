Contributing to JaxBo
=====================

We welcome contributions to JaxBo! This guide explains how to set up a development environment and contribute to the project.

Development Setup
-----------------

1. Fork and Clone
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/JaxBo.git
   cd JaxBo

2. Create Development Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   conda create -n jaxbo-dev python=3.10
   conda activate jaxbo-dev
   
   # Install in development mode
   pip install -e ".[dev]"

3. Install Pre-commit Hooks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pre-commit install

Code Style
----------

We use the following tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting  
- **flake8**: Linting
- **mypy**: Type checking

Run all checks:

.. code-block:: bash

   black jaxbo/
   isort jaxbo/
   flake8 jaxbo/
   mypy jaxbo/

Testing
-------

Run the test suite:

.. code-block:: bash

   pytest tests/

Add tests for new features in the ``tests/`` directory.

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

2. Make your changes and add tests

3. Ensure all tests pass and code follows style guidelines

4. Commit your changes:

   .. code-block:: bash

      git commit -m "Add my new feature"

5. Push to your fork and submit a pull request

Pull Request Guidelines
-----------------------

- Include a clear description of the changes
- Reference any related issues
- Add tests for new functionality
- Update documentation as needed
- Ensure CI passes

Types of Contributions
----------------------

We welcome various types of contributions:

- **Bug fixes**: Fix issues in existing code
- **New features**: Add new capabilities
- **Documentation**: Improve or expand documentation
- **Examples**: Add new examples or tutorials
- **Performance**: Optimize existing code
- **Tests**: Improve test coverage

Getting Help
------------

- Open an issue for questions or discussions
- Check existing issues before opening new ones
- Contact maintainers for major architectural changes

Thank you for contributing to JaxBo!
