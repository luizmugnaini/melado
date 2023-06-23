# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path configuration  -----------------------------------------------------
# In order for autodoc to work correctly, we have to set the path to project.
import sys
import os

sys.path.insert(0, os.path.abspath(".."))


# Generate the documentation automatically from docstrings
def run_apidoc(_):
    from sphinx.ext.apidoc import main

    parent_path = os.path.dirname(os.path.abspath(__file__))
    module_path = [os.path.join(parent_path, "..", "melado")]
    output_path = os.path.join(parent_path, "reference")
    main(["-e", "-M", "-o", output_path] + module_path)


def setup(app):
    app.connect("builder-inited", run_apidoc)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Melado"
copyright = "2023, Luiz G. Mugnaini A."
author = "Luiz G. Mugnaini A."
release = "0.1.0"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx.ext.viewcode",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_logo = "_static/logo.png"


# -- Extension configuration -------------------------------------------------

autodoc_member_order = "bysource"
