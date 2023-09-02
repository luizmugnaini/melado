# Melado 🍯

<p align="center">
    <img src="https://raw.githubusercontent.com/luizmugnaini/melado/master/docs/_static/logo.png" alt="Melado logo" width="300px"/>
</p>

> 🚧 WIP: This package is very immature and at its early stages. 🚧

Melado is a Python library designed to provide a comprehensive collection of machine learning algorithms
implemented using only the powerful numerical computing library, NumPy. Its
primary objective is to serve as a valuable learning resource for understanding
various machine learning algorithms without the complexity and performance
considerations associated with larger machine learning frameworks. You can find
more about melado in its [documentation](https://melado.readthedocs.io).

## Features

- Comprehensive collection of machine learning algorithms.
- Implemented using NumPy for efficient numerical computations.
- Easy-to-understand codebase for educational purposes.
- Simple and straightforward API.

## Installation

You can install Melado using pip:

```shell
pip install melado
```

## Usage

To use Melado, simply import the desired algorithm from the library and apply it to your data. Here's
an example of using the [weighted linear regression algorithm](https://en.wikipedia.org/wiki/Weighted_least_squares):

```python
from melado.linear import LinearRegression

# Prepare your data
X_train, y_train = ...
weights_train = ...

# Create a LinearRegression object
lin_reg = LinearRegression()

# Fit the data
lin_reg.fit(X_train, y_train, weights_train)

new_data, data_weights = ...

# Predict values from new data
labels = lin_reg.predict(new_data, data_weights)
```

<h2 id="development">
Development
</h2>

After cloning the repository, install [Poetry](https://python-poetry.org/) with your favorite package manager

```shell
paru -Syu python-poetry
```

Go to the `melado` directory and run the `start` script, which will install the virtual
environment using Poetry:

```shell
./start
```

The available `Makefile` can be used to either run the tests, run the
[Ruff](https://beta.ruff.rs/docs/) linter, run the [mypy](https://mypy-lang.org/)
typechecker, and build the docs with [Sphinx](https://www.sphinx-doc.org/):

```shell
make {tests,lint,docs}
```

## Contributing

We welcome contributions to Melado! If you'd like to contribute, please follow these guidelines:

- Fork the repository and clone it to your local machine. Read the [development](#development) section.
- Create a new branch for your feature or bug fix.
- Develop and test your code changes.
- Ensure that your code adheres to the existing coding style.
- Write clear and concise documentation for your feature.
- Submit a pull request explaining the changes you've made.

## License

Melado is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

If you have any questions, suggestions, or feedback, please me reach out at [luizmugnaini@gmail.com](mailto:luizmugnaini@gmail.com).

**Note:** Melado is currently in active development, and some features may be incomplete or subject to change.
We appreciate your understanding and encourage you to check for updates regularly.
