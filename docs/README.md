# JaxBo Documentation

This directory contains the ReadTheDocs-style documentation for JaxBo.

## Quick Start

To build the documentation locally:

```bash
# Install documentation dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html

# View documentation
open build/html/index.html
```

## Documentation Structure

- **source/**: Source files for documentation
  - **index.rst**: Main documentation page
  - **installation.rst**: Installation guide
  - **quickstart.rst**: Quick start guide
  - **api/**: API documentation
  - **tutorials/**: Step-by-step tutorials
  - **examples/**: Example galleries
  - **conf.py**: Sphinx configuration

- **build/**: Generated documentation (git-ignored)
- **requirements.txt**: Documentation dependencies
- **.readthedocs.yaml**: ReadTheDocs configuration

## Features

- **ReadTheDocs Theme**: Professional, responsive theme
- **API Documentation**: Automatically generated from docstrings
- **Cross-references**: Automatic linking between modules
- **Math Support**: LaTeX rendering with MathJax
- **Code Highlighting**: Syntax highlighting for code blocks
- **Search**: Full-text search capability
- **Multiple Formats**: HTML, PDF, ePub support

## Writing Documentation

### RST Format

Most documentation uses reStructuredText (RST):

```rst
Section Title
=============

Subsection
----------

**Bold text** and *italic text*.

.. code-block:: python

   # Code example
   import jaxbo
   
.. note::
   
   This is a note block.
```

### Markdown Support

Markdown is also supported via MyST parser:

```markdown
# Section Title

## Subsection

**Bold text** and *italic text*.

```python
# Code example
import jaxbo
```

```{note}
This is a note block.
```

### Docstrings

Use NumPy-style docstrings:

```python
def my_function(param1, param2=None):
    """
    Brief description.
    
    Parameters
    ----------
    param1 : type
        Description of param1.
    param2 : type, optional
        Description of param2.
        
    Returns
    -------
    type
        Description of return value.
    """
```

## Building Options

```bash
# Clean build
make clean html

# Build and serve locally
make html && python -m http.server 8000 -d build/html

# Check for broken links
make linkcheck

# Build PDF (requires LaTeX)
make latexpdf
```

## ReadTheDocs Integration

The documentation is configured for ReadTheDocs with:

- **.readthedocs.yaml**: Main configuration
- **requirements.txt**: Dependencies
- **pyproject.toml**: Package dependencies

## GitHub Actions

Automatic documentation building is set up via `.github/workflows/docs.yml`:

- Builds on every push/PR
- Deploys to GitHub Pages on main branch
- Uploads artifacts for preview

## Configuration

Key configuration in `source/conf.py`:

- **Theme**: `sphinx_rtd_theme`
- **Extensions**: autodoc, napoleon, intersphinx, etc.
- **Cross-references**: Links to NumPy, SciPy, JAX docs
- **Custom CSS**: Enhanced styling

## Contributing

When adding new modules or features:

1. Add docstrings following NumPy style
2. Update relevant RST files in `api/`
3. Add examples in `examples/`
4. Update tutorials if needed
5. Build locally to test: `make html`

## Troubleshooting

Common issues:

- **Import errors**: Ensure package is installed: `pip install -e .`
- **Missing dependencies**: Install docs deps: `pip install -e ".[docs]"`
- **Build warnings**: Check docstring formatting and cross-references
- **Theme issues**: Clear build cache: `make clean html`

For more help, see the [Sphinx documentation](https://www.sphinx-doc.org/).
