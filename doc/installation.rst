Installation
-------------------------------------------------------------------------------
BMLiNGAM is written in Python 2.7. If you are not familiar with Python, we recommend to use `Anaconda <https://store.continuum.io/cshop/anaconda/>`_, a distribution of Python environment with various packages provided by `Continuum Analytics <http://continuum.io>`_, for making installation of Python easier. 

BMLiNGAM will be installed in the Python directory. So ensure that the following environmental variables are appropriately set:

.. code-block:: console

    export ANACONDA_DIR=$HOME/anaconda # where anaconda is installed
    export PATH=$ANACONDA_DIR:$PATH
    export PATH=$ANACONDA_DIR/bin:$PATH
    export PYTHONPATH=$ANACONDA_DIR:$PATH

`git <https://git-scm.com>`_ is also required to install `PyMC3 <https://pymc-devs.github.io/pymc3/>`_, a Python package used by BMLiNGAM. 

To install BMLiNGAM, download the following wheel archive in any directory:

- :download:`BMLiNGAM-0.1.5-py2.py3-none-any.whl <../dist/BMLiNGAM-0.1.5-py2.py3-none-any.whl>`

Then, use pip (installed associated with Anaconda): 

.. code-block:: console

    $ pip install BMLiNGAM-0.1.5-py2.py3-none-any.whl

The following packages, required by BMLiNGAM, will also be automatically installed from `PyPi <https://pypi.python.org/pypi>`_ public package index: 

- numpy>=1.6.2
- scipy>=0.11
- matplotlib
- pandas

BMLiNGAM also requires `PyMC3 <https://pymc-devs.github.io/pymc3/>`_. Since it is not registered on PyPi, you need to install manually by following the instruction in the website. 

To uninstall BMLiNGAM:

.. code-block:: console

    $ pip uninstall bmlingam
