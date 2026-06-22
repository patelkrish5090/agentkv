"""
setup.py — legacy shim for editable installs (pip install -e .)

pyproject.toml is the authoritative project config.  This file exists only
because some older pip versions require it for editable installs of packages
that don't use a build backend that natively supports PEP 660.

If you add a native CUDA C++ extension (agentkv._C) in a future phase, move
the Extension() definition here and set AGENTKV_BUILD_CUDA=1 to enable it.
"""

from setuptools import setup

# All configuration lives in pyproject.toml.
# This call delegates entirely to setuptools' pyproject.toml reader.
setup()
