#! /usr/bin/env python
import setuptools  
from numpy.distutils.core import setup

descr = """Workflow objects for MEG (not yet) and EEG data processing"""

DISTNAME = 'MEEGbuddy'
DESCRIPTION = descr
MAINTAINER = 'Alex Rockhill'
MAINTAINER_EMAIL = 'arockhill@harvard.mgh.edu'
DOWNLOAD_URL = 'https://github.com/mghneurotherapeutics/meeg_resources.git'
VERSION = '0.0'

if __name__ == "__main__":
    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          long_description=open('README.md').read(),
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Microsoft :: Windows',
              'Operating System :: POSIX',
              'Operating System :: Unix',
              'Operating System :: MacOS',
          ],
          platforms='any',
          packages=[
              'MEEGbuddy'
          ],
          )
