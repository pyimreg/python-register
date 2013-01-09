#! /usr/bin/env python

DISTNAME = 'imreg'
DESCRIPTION = 'Image registration toolkit'
LONG_DESCRIPTION = """
"imreg" is an image registration package for python that makes it easy to
automatically align image data.
"""
MAINTAINER = 'Nathan Faggian, Riaan Van Den Dool, Stefan Van Der Walt'
MAINTAINER_EMAIL = 'nathan.faggian@gmail.com'
URL = 'pyimreg.github.com'
LICENSE = 'Apache License (2.0)'
DOWNLOAD_URL = ''
VERSION = '0.1'

import os
import setuptools
from numpy.distutils.core import setup
try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'): os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)

    config.set_options(
            ignore_setup_xxx_py=True,
            assume_default_configuration=True,
            delegate_options_to_subpackages=True,
            quiet=True)

    config.add_subpackage('imreg')

    return config

if __name__ == "__main__":
    setup(
        name=DISTNAME,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        url=URL,
        license=LICENSE,
        download_url=DOWNLOAD_URL,
        version=VERSION,

        classifiers =
            [ 'Development Status :: 4 - Beta',
              'Environment :: Console',
              'Intended Audience :: Developers',
              'Intended Audience :: Science/Research',
              'License :: OSI Approved :: Apache Software License',
              'Topic :: Scientific/Engineering'],

        configuration=configuration,
        install_requires=[],
        packages=setuptools.find_packages(),
        include_package_data=True,
        zip_safe=False,
        cmdclass={'build_py': build_py},
        )
