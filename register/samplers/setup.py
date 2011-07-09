from distutils.core import setup
from distutils.extension import Extension

import numpy 

setup(
      name="libsampler",
      ext_modules = [ Extension("libsampler", 
                                ["libsampler.cpp"], 
                                include_dirs=[numpy.get_include()])
                    ]
     )
