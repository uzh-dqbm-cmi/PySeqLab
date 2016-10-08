'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''
from distutils.core import setup
import os
import sys

# setup requirements
if sys.version_info < (3,4,0):
    sys.exit('Python 3.4.0 or above is required.\n')

try:
    from setuptools import setup
except ImportError:
    sys.exit('setuptools is missing -- please install.\n')

try:
    from pip.req import parse_requirements
except ImportError:
    sys.exit('pip is missing -- please install.\n')



# package description
expl_files = ['README.md']
long_description = '\n'.join([open(f, 'r').read() for f in expl_files])


setup(name='pyseqlab',
      packages=['pyseqlab'],
      package_dir = {"pyseqlab":"pyseqlab"},
      version="1.0",
      description='A package for performing structured prediction (i.e.sequence labelling and segmentation learning.',
      long_description=long_description,
      author="Ahmed Allam",
      author_email='ahmed.allam@yale.edu',
      license="",
      url='',
      download_url='https://bitbucket.org/A_2/pyseqlab/downloads',
      keywords='conditional random field, structured prediction, bioinformatics',
      install_requires=["scipy>=0.13"],
      scripts=install_scripts,
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'Natural Language :: English',
                   'Operating System :: OS Independent',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Information Analysis'])