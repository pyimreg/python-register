.. -*- mode: rst -*-

About
=====

python-register is a python module for image registration built ontop of scipy and numpy.

It is currently maintained by Nathan Faggian.

Important links
===============

- Official source code: https://github.com/nfaggian/python-register

Dependencies
============

The required dependencies to build the software are python >= 2.5,
setuptools, NumPy >= 1.5, SciPy >= 0.9 and a working C++ compiler.

To run the tests you will also need py.test >= 2.0.


Install
=======

This packages uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --home

To install for all users on Unix/Linux::

  python setup.py build
  
  sudo python setup.py install


Development
===========

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/nfaggian/python-regsiter.git
