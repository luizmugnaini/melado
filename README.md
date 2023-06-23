# Melado 🍯

<p align="center">
    <img scr="docs/_static/mini_logo.png" alt="Melado logo" width="300px"/>
</p>

Melado is a Python library, although very immature and at its early stages,
designed to provide a comprehensive collection of machine learning algorithms
implemented using only the powerful numerical computing library, NumPy. Its
primary objective is to serve as a valuable learning resource for understanding
various machine learning algorithms without the complexity and performance
considerations associated with larger machine learning frameworks.

## Development

After cloning the repository, install [Poetry](https://python-poetry.org/)
with your favorite package manager:

```shell
paru -Syu python-poetry
```

Go to the `melado` directory and install the dependencies:

```shell
poetry install
```

The available `Makefile` can be used to either run the tests, run the
[Ruff](https://beta.ruff.rs/docs/) linter, run the [Mypy](https://mypy-lang.org/)
typechecker, and build the docs with [Sphinx](https://www.sphinx-doc.org/):

```shell
make {tests,lint,typecheck,docs}
```
