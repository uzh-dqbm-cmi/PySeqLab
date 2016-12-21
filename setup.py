'''
@author: ahmed allam <ahmed.allam@yale.edu>
'''

from setuptools import setup


# package description
# to change into .rst file
expl_files = ['README.md']
long_description = '\n'.join([open(f, 'r').read() for f in expl_files])

if __name__ == "__main__":
    setup(name='PySeqLab',
          version="1.3.0",
          description='A package for performing structured prediction (i.e.sequence labeling and segmentation).',
          long_description=long_description,
          author="Ahmed Allam",
          author_email='ahmed.allam@yale.edu',
          license="",
          url='https://bitbucket.org/A_2/pyseqlab/',
          download_url='https://bitbucket.org/A_2/pyseqlab/downloads',
          keywords='conditional random fields, structured prediction, bioinformatics',
          packages=["pyseqlab"],
          install_requires=["numpy", "scipy>=0.13"],
          classifiers=['Development Status :: 4 - Beta',
                       'Environment :: Console',
                       'Intended Audience :: Science/Research',
                       'Natural Language :: English',
                       'Operating System :: OS Independent',
                       'Programming Language :: Python :: 3',
                       'Programming Language :: Python :: 3.4',
                       'Programming Language :: Python :: 3.5',
                       'Topic :: Scientific/Engineering :: Bio-Informatics',
                       'Topic :: Scientific/Engineering :: Information Analysis'])
    
