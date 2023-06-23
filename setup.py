import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="melado",
    version="0.1.0",
    author="Luiz Gustavo Mugnaini Anselmo",
    author_email="luizmugnaini@gmail.com",
    url="https://github.com/luizmugnaini/melado",
    description="NumPy-based machine learning library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "poetry",
    ],
    extra_require={
        "linting": ["mypy", "ruff", "black"],
        "testing": ["pytest", "scikit-learn"],
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
