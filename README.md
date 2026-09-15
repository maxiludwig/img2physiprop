# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/maxiludwig/img2physiprop/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                     |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/i2pp/\_\_init\_\_.py                                                 |        0 |        0 |    100% |           |
| src/i2pp/core/\_\_init\_\_.py                                            |        0 |        0 |    100% |           |
| src/i2pp/core/configuration\_validator/validation\_helpers.py            |        6 |        0 |    100% |           |
| src/i2pp/core/configuration\_validator/validator.py                      |      123 |        8 |     93% |130, 132, 134, 184, 242-244, 258 |
| src/i2pp/core/discretization\_helpers.py                                 |       56 |        0 |    100% |           |
| src/i2pp/core/discretization\_readers/discretization\_format.py          |       11 |        0 |    100% |           |
| src/i2pp/core/discretization\_readers/discretization\_reader.py          |       39 |        1 |     97% |       168 |
| src/i2pp/core/discretization\_readers/lnmmeshio\_reader.py               |       79 |        3 |     96% |71-72, 147 |
| src/i2pp/core/discretization\_readers/trimesh\_reader.py                 |       20 |        3 |     85% |41, 69, 77 |
| src/i2pp/core/export\_data.py                                            |       36 |        3 |     92% | 45, 59-62 |
| src/i2pp/core/exporters/discretization\_exporter.py                      |       54 |        0 |    100% |           |
| src/i2pp/core/exporters/export\_format.py                                |       15 |        0 |    100% |           |
| src/i2pp/core/exporters/exporter.py                                      |       17 |        1 |     94% |        38 |
| src/i2pp/core/exporters/json\_exporter.py                                |       27 |        0 |    100% |           |
| src/i2pp/core/exporters/txt\_exporter.py                                 |       12 |        0 |    100% |           |
| src/i2pp/core/image\_readers/dicom\_reader.py                            |       53 |        5 |     91% |85, 122-123, 148, 151 |
| src/i2pp/core/image\_readers/image\_format.py                            |       34 |        1 |     97% |        85 |
| src/i2pp/core/image\_readers/image\_reader.py                            |       64 |        4 |     94% |47, 66, 240, 264 |
| src/i2pp/core/image\_readers/label\_map.py                               |       35 |        0 |    100% |           |
| src/i2pp/core/image\_readers/nifti\_reader.py                            |       48 |        1 |     98% |        77 |
| src/i2pp/core/image\_readers/png\_reader.py                              |       57 |        1 |     98% |       193 |
| src/i2pp/core/import\_image.py                                           |       41 |        1 |     98% |       141 |
| src/i2pp/core/interpolate\_element\_data.py                              |       29 |        1 |     97% |        28 |
| src/i2pp/core/interpolators/interpolator.py                              |       37 |        2 |     95% |  115, 191 |
| src/i2pp/core/interpolators/interpolator\_all\_voxel.py                  |      105 |        7 |     93% |202-203, 240-242, 330, 410 |
| src/i2pp/core/interpolators/interpolator\_center.py                      |       25 |        1 |     96% |       103 |
| src/i2pp/core/interpolators/interpolator\_label\_map.py                  |       70 |        1 |     99% |       199 |
| src/i2pp/core/interpolators/interpolator\_nodes.py                       |       72 |       20 |     72% |71, 112, 116, 125, 130, 137-140, 145-146, 174-195, 229, 270 |
| src/i2pp/core/interpolators/interpolator\_types.py                       |       30 |        0 |    100% |           |
| src/i2pp/core/run.py                                                     |       41 |        1 |     98% |        59 |
| src/i2pp/core/transform\_data.py                                         |        8 |        0 |    100% |           |
| src/i2pp/core/user\_function\_transformer/user\_function\_transformer.py |       36 |        1 |     97% |        49 |
| src/i2pp/core/utilities.py                                               |       80 |        5 |     94% |58, 60, 333, 337-338 |
| src/i2pp/core/visualize\_results.py                                      |       42 |       18 |     57% |32-38, 71-79, 88-96, 127-133, 143-149 |
| src/i2pp/core/visualizers/discretization\_visualizer.py                  |       16 |        0 |    100% |           |
| src/i2pp/core/visualizers/image\_visualizer.py                           |       19 |        0 |    100% |           |
| src/i2pp/core/visualizers/visualizer.py                                  |       96 |       12 |     88% |59, 79, 156-169, 198-209, 236, 293, 307-308 |
| src/i2pp/main.py                                                         |       16 |        0 |    100% |           |
| **TOTAL**                                                                | **1549** |  **101** | **93%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/maxiludwig/img2physiprop/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/maxiludwig/img2physiprop/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/maxiludwig/img2physiprop/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/maxiludwig/img2physiprop/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fmaxiludwig%2Fimg2physiprop%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/maxiludwig/img2physiprop/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.