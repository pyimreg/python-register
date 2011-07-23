import os

def configuration(parent_package='', top_path=None):

    from numpy.distutils.misc_util import Configuration

    config = Configuration('register', parent_package, top_path)

    config.add_subpackage('models')
    config.add_subpackage('metrics')
    config.add_subpackage('samplers')
    config.add_subpackage('visualize')

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    config = configuration(top_path='').todict()
    setup(**config)
