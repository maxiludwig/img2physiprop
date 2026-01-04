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

The remaining parts of the README are structured as follows:

- [Installation](#installation)
- [Execution](#execution)
  - [Execute img2physiprop](#execute-img2physiprop)
  - [Run testing framework and create coverage report](#run-testing-framework-and-create-coverage-report)
  - [Create documentation](#create-documentation)
  - [Interpolation and value settings](#interpolation-and-value-settings)
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

To execute img2physiprop run

```
i2pp --config path/to/config.yaml
```

with your custom configuration file. A template configuration file containing all possible input configurations can be found in the folder `templates/config`.

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

### Interpolation and value settings

- Interpolation methods (processing.interpolation.method):
  - nodes: Interpolates values at the element’s nodes and assigns the element mean (ignoring NaN nodes). Fast and robust; respects node sampling.
  - nodes_weighted: Like `nodes`, but computes a weighted mean using node-specific weights (`dis.nodes.weights`), which are set via `processing.interpolation.node_weight.surface` and `processing.interpolation.node_weight.interior`. This reduces the influence of low-weight nodes.
  - elementcenter: Interpolates at each element centroid and assigns that value. Reliable fallback if no voxels lie inside the element.
  - allvoxels: Collects all voxels whose grid coordinates lie inside the convex hull of the element nodes; assigns the mean value; optionally filters outliers.
  - allvoxels_weighted: Computes a voxel-weighted mean where voxel weights derive from node weights and inverse node-to-voxel distances; optionally filters outliers.

- Element and node value overrides:
  - set_surface_node_value: If provided, all nodes that belong to any surface receive the fixed value (vector size must match the number of pixel channels); only relevant for `nodes`and `nodes_weighted` interpolation method.
  - set_surface_element_value: If provided, all elements touching any surface node receive the fixed value (scalar or vector); applicable to all interpolation methods.
  - When both are set, surface node override is ignored (element override wins).

- Outlier filtering (processing.interpolation.filter_outliers):
  - In `allvoxels` and `allvoxels_weighted`, if enabled and enough voxels are present (>5), outliers are removed using a modified Z-score (median/MAD-based, threshold=3.5) before averaging.

- Fallbacks and warnings:
  - If outlier filtering removes all voxels, the method falls back to the unfiltered mean.
  - If an element contains no voxels (allvoxels modes), interpolation falls back to the element center.
  - If interpolated points fall outside the image grid, element data is NaN and a warning summary is logged after processing.

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
