#! /usr/bin/env python

descr   = """Place long description here.

"""

DISTNAME            = 'python-register'
DESCRIPTION         = 'Image registration toolbox for SciPy'
LONG_DESCRIPTION    = descr
MAINTAINER          = 'Nathan Faggian'
MAINTAINER_EMAIL    = 'nathan.faggian@gmail.com'
URL                 = ''
LICENSE             = 'Apache License (2.0)'
DOWNLOAD_URL        = ''
VERSION             = '0.1'

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

    config.add_subpackage('register')
#    config.add_subpackage(DISTNAME)

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
        zip_safe=False, # the package can run out of an .egg file

        cmdclass={'build_py': build_py},
        )
