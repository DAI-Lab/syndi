.. raw:: html

   <p align="left">
   <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt="DAI-Lab" />
   <i>An open source project from Data to AI Lab at MIT.</i>
   </p>

|Development Status| |PyPi Shield| |Run Tests Shield|
|Downloads|

Syndi
=====

e2e pipeline to generate synthetic data for empowering machine learning models

Overview
--------

TODO: Provide a short overview of the project here.

Install
-------

Requirements
~~~~~~~~~~~~

**syndi** has been developed and tested on [Python 3.6 and 3.7](https://www.python.org/downloads/)

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **syndi** is run.

These are the minimum commands needed to create a virtualenv using python3.6 for **syndi**:

```bash
pip install virtualenv
virtualenv -p $(which python3.6) syndi-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source syndi-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **syndi**!

<!-- Uncomment this section after releasing the package to PyPI for installation instructions
## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **syndi**:

```bash
pip install syndi
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

Install from source
~~~~~~~~~~~~~~~~~~~

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:DAI-Lab/syndi.git
cd syndi
git checkout stable
make install
```

Install for Development
~~~~~~~~~~~~~~~~~~~~~~~

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://DAI-Lab.github.io/syndi/contributing.html#get-started)
for more details about this process.

pyenv installation support can be found [here](https://stackoverflow.com/questions/66482346/problems-installing-python-3-6-with-pyenv-on-mac-os-big-sur)

Use the following pyenv [documentation](https://github.com/pyenv/pyenv#basic-github-checkout) for setting up pyenv

Setting up tox:
```
pip install tox tox-pyenv
pyenv install ...
pyenv local ...
```
Where ... should be versions of python 3.6, 3.7



Quickstart
----------

In this short tutorial we will guide you through a series of steps that will help you
getting started with **syndi**.

TODO: Create a step by step guide here.

What's next?
------------

For more details about **syndi** and all its possibilities
and features, please check the [documentation site](
https://DAI-Lab.github.io/syndi/).


Explore Syndi
-------------

* `Getting Started <getting_started/index.html>`_
* `User Guides <user_guides/index.html>`_
* `API Reference <api_reference/index.html>`_
* `Developer Guides <developer_guides/index.html>`_
* `Release Notes <history.html>`_

--------------

.. |Development Status| image:: https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow
   :target: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
.. |PyPi Shield| image:: https://img.shields.io/pypi/v/orion-ml.svg
   :target: https://pypi.python.org/pypi/orion-ml
.. |Run Tests Shield| image:: https://github.com/sintel-dev/Orion/workflows/Run%20Tests/badge.svg
   :target: https://github.com/sintel-dev/Orion/actions?query=workflow%3A%22Run+Tests%22+branch%3Amaster
.. |Downloads| image:: https://pepy.tech/badge/orion-ml
   :target: https://pepy.tech/project/orion-ml
.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/sintel-dev/Orion/master?filepath=notebooks


.. toctree::
    :maxdepth: 3
    :hidden:
    :titlesonly:

    getting_started/index
    user_guides/index
    api_reference/index
    developer_guides/index
    Release Notes <history>
    authors

.. _Data to AI Lab at MIT: https://dai.lids.mit.edu/
.. _part 1: https://t.co/yIFVM1oRwQ?amp=1
.. _part 2: https://link.medium.com/cGsBD0Fevbb
.. _part 3: https://link.medium.com/FqCrFXMevbb
