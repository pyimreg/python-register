.. -*- mode: rst -*-

About
=====

python-register is a python module for image registration built ontop of scipy and numpy.

It is currently maintained by Nathan Faggian, Riaan Van Den Dool and Stefan Van Der Walt.

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

Basic rules for commits to the python-register repository:

 + master is our stable "release" branch.
	
 + feature branches (or contributor pull requests) for each ticket on github are merged (into master) after review only. 
 
 + tests for new features using py.test *must* exist before merges.

GIT
~~~

You can check the latest sources with the command::

    git clone git://github.com/nfaggian/python-regsiter.git
    
Contributors
~~~~~~~~~~~~~

Follow: Fork + Pull Model::
     
    http://help.github.com/send-pull-requests/

Maintainers
~~~~~~~~~~~~~

Follow: Shared Repository Model

Tracking an already formed branch:

   git checkout -b localBranch origin/remoteBranch

Forming a new branch and pushing to github:
   
   git checkout -b localFeature
	
   git push origin localFeature

