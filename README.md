# Fake Convergence Beam Electron Diffraction (FakeCBED)

[![Test library](https://github.com/mrfitzpa/fakecbed/actions/workflows/test_library.yml/badge.svg)](https://github.com/mrfitzpa/fakecbed/actions/workflows/test_library.yml)
[![Code Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/mrfitzpa/adb03c4c54f978f44822ffa033fe6762/raw/fakecbed_coverage_badge.json)](https://github.com/mrfitzpa/fakecbed/actions/workflows/measure_code_coverage.yml)
[![Documentation](https://img.shields.io/badge/docs-read-brightgreen)](https://mrfitzpa.github.io/fakecbed)
[![PyPi Version](https://img.shields.io/pypi/v/fakecbed.svg)](https://pypi.org/project/fakecbed)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/fakecbed.svg)](https://anaconda.org/conda-forge/fakecbed)
[![License](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-10.5281/zenodo.18297325-blue.svg)](https://doi.org/10.5281/zenodo.18297325)

`fakecbed` is a Python library for generating quickly images that imitate
convergent beam electron diffraction patterns.

Visit the `fakecbed` [website](https://mrfitzpa.github.io/fakecbed) for a web
version of the installation instructions, the reference guide, and the examples
archive.

The source code can be found in the [`fakecbed` GitHub
repository](https://github.com/mrfitzpa/fakecbed).



## Table of contents

- [Instructions for installing and uninstalling
  `fakecbed`](#instructions-for-installing-and-uninstalling-fakecbed)
  - [Installing `fakecbed`](#installing-fakecbed)
    - [Installing `fakecbed` using `pip`](#installing-fakecbed-using-pip)
    - [Installing `fakecbed` using
      `conda`](#installing-fakecbed-using-conda)
  - [Uninstalling `fakecbed`](#uninstalling-fakecbed)
- [Learning how to use `fakecbed`](#learning-how-to-use-fakecbed)



## Instructions for installing and uninstalling `fakecbed`



### Installing `fakecbed`

For all installation scenarios, first open up the appropriate command line
interface. On Unix-based systems, you could open e.g. a terminal. On Windows
systems you could open e.g. an Anaconda Prompt as an administrator.

Before installing `fakecbed`, it is recommended that users install `PyTorch` in
the same environment that they intend to install `fakecbed` according to the
instructions given [here](https://pytorch.org/get-started/locally/) for their
preferred PyTorch installation option.



#### Installing `fakecbed` using `pip`

Before installing `fakecbed`, make sure that you have activated the (virtual)
environment in which you intend to install said package. After which, simply run
the following command:

    pip install fakecbed

The above command will install the latest stable version of `fakecbed`.

To install the latest development version from the main branch of the [fakecbed
GitHub repository](https://github.com/mrfitzpa/fakecbed), one must first clone
the repository by running the following command:

    git clone https://github.com/mrfitzpa/fakecbed.git

Next, change into the root of the cloned repository, and then run the following
command:

    pip install .

Note that you must include the period as well. The above command executes a
standard installation of `fakecbed`.

Optionally, for additional features in `fakecbed`, one can install additional
dependencies upon installing `fakecbed`. To install a subset of additional
dependencies (along with the standard installation), run the following command
from the root of the repository:

    pip install .[<selector>]

where `<selector>` can be one of the following:

* `tests`: to install the dependencies necessary for running unit tests;
* `examples`: to install the dependencies necessary for executing files stored
  in `<root>/examples`, where `<root>` is the root of the repository;
* `docs`: to install the dependencies necessary for documentation generation;
* `all`: to install all of the above optional dependencies.

Alternatively, one can run:

    pip install fakecbed[<selector>]

elsewhere in order to install the latest stable version of `fakecbed`, along
with the subset of additional dependencies specified by `<selector>`.



#### Installing `fakecbed` using `conda`

Before proceeding, make sure that you have activated the (virtual) `conda`
environment in which you intend to install said package. For Windows systems,
users must install `PyTorch` separately prior to following the remaining
instructions below.

To install `fakecbed` using the `conda` package manager, run the following
command:

    conda install -c conda-forge fakecbed

The above command will install the latest stable version of `fakecbed`.



### Uninstalling `fakecbed`

If `fakecbed` was installed using `pip`, then to uninstall, run the following
command:

    pip uninstall fakecbed

If `fakecbed` was installed using `conda`, then to uninstall, run the following
command:

    conda remove fakecbed



## Learning how to use `fakecbed`

For those new to the `fakecbed` library, it is recommended that they take a look
at the [Examples](https://mrfitzpa.github.io/fakecbed/examples.html) page, which
contain code examples that show how one can use the `fakecbed` library. While
going through the examples, readers can consult the [fakecbed reference
guide](https://mrfitzpa.github.io/fakecbed/_autosummary/fakecbed.html) to
understand what each line of code is doing.