<h1 align="center">
  img2physiprop
</h1>

<div align="center">

[![Pipeline](https://github.com/maxiludwig/img2physiprop/actions/workflows/main_pipeline.yml/badge.svg)](https://github.com/maxiludwig/img2physiprop/actions/workflows/main_pipeline.yml)
[![Documentation](https://github.com/maxiludwig/img2physiprop/actions/workflows/main_documentation.yml/badge.svg)](https://maxiludwig.github.io/img2physiprop/)
[![Coverage badge](https://github.com/maxiludwig/img2physiprop/raw/python-coverage-comment-action-data/badge.svg)](https://github.com/maxiludwig/img2physiprop/tree/python-coverage-comment-action-data)

</div>

img2physiprop (Image to Physical Property) is a python package that maps medical image data to physical properties. This makes it possible to vary e.g. material parameters in FE simulations according to patient specific medical image data. The package includes the following features to ease the development process and ensure a high code quality:

- [PyTest](https://docs.pytest.org/) testing framework including an enforced minimum coverage check
- Automated [Github CI/CD](https://resources.github.com/devops/ci-cd/)
- Exhaustive [Pre-Commit](https://pre-commit.com) framework to automatically check code formatting and code quality
- Automatically generated [Documentation](https://pdoc.dev) based on the included Python docstrings
- Pre-defined framework to gather global settings (see [`main_example_config.yaml`](./src/i2pp/main_example_config.yaml)) and execute a specific workflow

The remaining parts of the readme are structured as follows:

- [Installation](#installation)
- [Execution](#execution)
  - [Execute img2physiprop](#execute-img2physiprop)
  - [Run testing framework and create coverage report](#run-testing-framework-and-create-coverage-report)
  - [Create documentation](#create-documentation)
- [Dependency Management](#dependency-management)
- [Contributing](#contributing)
- [License](#license)



## Installation

For a quick and easy start an Anaconda/Miniconda environment is highly recommended. Other ways to install img2physiprop are possible but here the installation procedure is explained based on a conda install. After installing Anaconda/Miniconda
execute the following steps:

- Create a new Anaconda environment based on the [`environment.yml`](./environment.yml) file:
```
conda env create -f environment.yml
```

- Activate your newly created environment:
```
conda activate i2pp
```

- Initialize all submodules
```
git submodule update --init --recursive
```

- All necessary third party libraries for all submodules can be installed using:
```
git submodule --quiet foreach --recursive pip install -e .
```

- Install all img2physiprop requirements with:
```
pip install -e .
```

- Finally, install the pre-commit hook with:
```
pre-commit install
```

Now you are up and running 🎉

## Execution

### Execute img2physiprop

To execute img2physiprop either run

```
i2pp
````

to execute img2physiprop with the provided exemplary config or use

```
i2pp --config_file_path ../path/to/config.yaml
````

to utilize your own externally provided config file. Therein, all necessary configurations can be found.

### Run testing framework and create coverage report

To locally execute the tests and create the html coverage report simply run

```
pytest
```

### Create documentation

To locally create the documentation from the provided docstrings simply run

```
pdoc --html --output-dir docs src/i2pp
```

## Dependency Management

To ease the dependency update process [`pip-tools`](https://github.com/jazzband/pip-tools) is utilized. To create the necessary [`requirements.txt`](./requirements.txt) file simply execute

```
pip-compile --all-extras --output-file=requirements.txt requirements.in
````

To upgrade the dependencies simply execute

```
pip-compile --all-extras --output-file=requirements.txt --upgrade requirements.in
````

Finally, perforfmance critical packages such as Numpy and Numba are installed via conda to utilize BLAS libraries.

## Contributing

All contributions are welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md) for more information.

## License

This project is licensed under a MIT license. For further information check [`LICENSE.md`](./LICENSE.md).
