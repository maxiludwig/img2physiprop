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

- Interpolation methods (`processing.interpolation.method`):
  - `nodes`: Interpolates values at the element’s nodes and assigns the element mean (ignoring NaN nodes). Fast and robust; respects node sampling.
  - `nodes_scaled`: Like `nodes`, but computes a scaled mean using node-specific scaling factors (`dis.nodes.scaling_factors`), which are set via `processing.interpolation.node_scaling_factors.surface` and `processing.interpolation.node_scaling_factors.interior`. This adjusts the influence of specific nodes.
  - `elementcenter`: Interpolates at each element centroid and assigns that value.
  - `allvoxels`: Collects all voxels whose grid coordinates lie inside the convex hull of the element nodes; assigns the mean value; optionally filters outliers.
  - `allvoxels_scaled`: Computes a voxel-weighted mean where voxel weights derive from node scaling factors and inverse node-to-voxel distances. The influence of the distance (decay) is controlled by `processing.interpolation.inverse_distance_power` p (p=1 linear, p=2 quadratic (default), p>=3 increasingly like step function); Optionally filters outliers.

- Element and node value overrides:
  - `processing.interpolation.set_surface_node_value`: If provided, all nodes that belong to any surface receive the fixed value (vector size must match the number of pixel channels); only relevant for `nodes` and `nodes_scaled` interpolation methods.
  - `processing.interpolation.set_surface_element_value`: If provided, all elements touching any surface node receive the fixed value (scalar or vector); applicable to all interpolation methods.

- Outlier filtering (`processing.interpolation.filter_outliers`):
  - In `allvoxels` and `allvoxels_scaled`, if enabled and enough voxels are present (>5), outliers are removed using a modified Z-score (median/MAD-based, threshold=3.5) before averaging.

- Fallbacks and warnings:
  - If outlier filtering removes all voxels, the method falls back to the unfiltered mean.
  - If an element contains no voxels (allvoxels modes), interpolation falls back to the element center.
  - If interpolated points fall outside the image grid, element data is NaN and a warning summary is logged after processing.

### Discretization formats and element filtering

- Supported discretization formats: `.4C.yaml`, `.mesh` and `.vtu`. `.vtu` files are read via `lnmmeshio`, which also supports meshes consisting of line2 elements (e.g. airway trees). For line2 elements, `nodes` averages the two node values and `elementcenter` samples at the midpoint of the line.
- Element filtering (`import.discretization.options`):
  - `element_filter`: Only elements whose field `field` has one of the given `values` are processed. The field is looked up in the element data (e.g. a VTU cell data array such as `block_id`) and in the element options (e.g. the material `MAT`), e.g. `element_filter: {field: block_id, values: [2]}` or `element_filter: {field: MAT, values: [1, 2]}`.
  - `material_ids` (deprecated): Translated into `element_filter: {field: MAT, values: material_ids}` with a warning. Cannot be combined with `element_filter`.
- Export into the input discretization (`export.type: discretization`, only for `.vtu` discretizations): writes `{file_name}.vtu`, a copy of the complete input mesh including all of its point and cell data, with the results as cell data. The user function must return a structured array with the field `index` and one or more numeric value fields; each value field is written to the cell data field of the same name, e.g. the fields `E` and `nu` become the cell data fields `E` and `nu` (vector fields become multi-component fields). `output_parameter_name` is only used for the `json` export and must not be set otherwise. Unselected elements are NaN. If the input mesh already contains the field (e.g. the output of a previous run on another element selection), only the selected elements are overwritten. This allows running i2pp consecutively on different element selections, e.g. first on the terminal units and then on the airways of a lung tree. The input file itself is never overwritten.
- Note: the element IDs in the `json`/`txt` export refer to the numbering of the *filtered* discretization.

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
