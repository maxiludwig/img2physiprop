# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/maxiludwig/img2physiprop/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                    |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| src/i2pp/\_\_init\_\_.py                                                |        0 |        0 |    100% |           |
| src/i2pp/core/\_\_init\_\_.py                                           |        0 |        0 |    100% |           |
| src/i2pp/core/discretization\_helpers.py                                |       68 |        0 |    100% |           |
| src/i2pp/core/discretization\_reader\_classes/discretization\_reader.py |       30 |        1 |     97% |       126 |
| src/i2pp/core/discretization\_reader\_classes/fourc\_yaml\_reader.py    |       43 |        1 |     98% |        99 |
| src/i2pp/core/discretization\_reader\_classes/mesh\_reader.py           |       18 |        3 |     83% |33, 58, 66 |
| src/i2pp/core/export\_data.py                                           |       99 |        4 |     96% |81, 86, 235, 264 |
| src/i2pp/core/image\_reader\_classes/dicom\_reader.py                   |       53 |        5 |     91% |85, 122-123, 148, 151 |
| src/i2pp/core/image\_reader\_classes/image\_reader.py                   |       72 |        4 |     94% |49, 68, 242, 266 |
| src/i2pp/core/image\_reader\_classes/png\_reader.py                     |       57 |        1 |     98% |       193 |
| src/i2pp/core/import\_image.py                                          |       65 |        4 |     94% |57, 71, 86, 206 |
| src/i2pp/core/interpolate\_element\_data.py                             |       21 |        7 |     67% |60-69, 100-106 |
| src/i2pp/core/interpolator\_classes/interpolator.py                     |       27 |        4 |     85% |115, 121, 127, 153 |
| src/i2pp/core/interpolator\_classes/interpolator\_all\_voxel.py         |       47 |        9 |     81% |   186-214 |
| src/i2pp/core/interpolator\_classes/interpolator\_center.py             |       25 |        1 |     96% |       103 |
| src/i2pp/core/interpolator\_classes/interpolator\_nodes.py              |       18 |        1 |     94% |        76 |
| src/i2pp/core/run.py                                                    |       34 |        6 |     82% |54, 62-65, 88 |
| src/i2pp/core/utilities.py                                              |       29 |        2 |     93% |    57, 59 |
| src/i2pp/core/visualization\_classes/discretization\_visualization.py   |       14 |        0 |    100% |           |
| src/i2pp/core/visualization\_classes/image\_visualization.py            |       19 |        0 |    100% |           |
| src/i2pp/core/visualization\_classes/visualization.py                   |       95 |       12 |     87% |59, 79, 154-167, 196-207, 234, 291, 305-306 |
| src/i2pp/core/visualize\_results.py                                     |       42 |       18 |     57% |32-38, 74-82, 91-99, 130-136, 146-152 |
| src/i2pp/main.py                                                        |       16 |        0 |    100% |           |
|                                                               **TOTAL** |  **892** |   **83** | **91%** |           |


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