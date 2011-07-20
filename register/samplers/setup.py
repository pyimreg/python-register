#!/usr/bin/env python

import os

base_path = os.path.abspath(os.path.dirname(__file__))

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import \
         Configuration, get_numpy_include_dirs

    config = Configuration('samplers', parent_package, top_path)

    config.add_extension('libsampler', sources=['libsampler.cpp'],
                         include_dirs=[get_numpy_include_dirs()])

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup

    setup(
        **(configuration(top_path='').todict())
    )
