.. _installation_instructions_sec:

Installation instructions
=========================

Installing fakecbed
-------------------

For all installation scenarios, first open up the appropriate command line
interface. On Unix-based systems, you would open a terminal. On Windows systems
you would open an Anaconda Prompt as an administrator.

Before installing ``fakecbed``, it is recommended that users install ``PyTorch``
in the same environment that they intend to install ``fakecbed`` according to
the instructions given `here <https://pytorch.org/get-started/locally/>`_ for
their preferred PyTorch installation option.

Installing fakecbed using pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install ``fakecbed`` using ``pip`` is to run the following
command::

  pip install fakecbed

The above command will install the latest stable version of ``fakecbed``.

To install the latest development version from the main branch of the `fakecbed
GitHub repository <https://github.com/mrfitzpa/fakecbed>`_, one must first clone
the repository by running the following command::

  git clone https://github.com/mrfitzpa/fakecbed.git

Next, change into the root of the cloned repository, and then run the following
command::

  pip install .

Note that you must include the period as well. The above command executes a
standard installation of ``fakecbed``.

Optionally, for additional features in ``fakecbed``, one can install additional
dependencies upon installing ``fakecbed``. To install a subset of additional
dependencies (along with the standard installation), run the following command
from the root of the repository::

  pip install .[<selector>]

where ``<selector>`` can be one of the following:

* ``tests``: to install the dependencies necessary for running unit tests;
* ``examples``: to install the dependencies necessary for running the jupyter
  notebooks stored in ``<root>/examples``, where ``<root>`` is the root of the
  repository;
* ``docs``: to install the dependencies necessary for documentation generation;
* ``all``: to install all of the above optional dependencies.

Installing fakecbed using conda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For Windows systems, users must install ``PyTorch`` separately prior to
following the remaining instructions below.

To install ``fakecbed`` using the ``conda`` package manager, run the following
command::

  conda install -c conda-forge fakecbed

The above command will install the latest stable version of ``fakecbed``.

Uninstalling fakecbed
---------------------

If ``fakecbed`` was installed using ``pip``, then to uninstall, run the
following command from the root of the repository::

  pip uninstall fakecbed

If ``fakecbed`` was installed using ``conda``, then to uninstall, run the
following command from the root of the repository::

  conda remove fakecbed
