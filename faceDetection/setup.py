#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from setuptools import setup, setuptools
from faceDetection import __version__

def readme():
    with open('README.rst', encoding="UTF-8") as f:
        return f.read()
if sys.version_info < (3, 4, 1):
    sys.exit('Python version verify')
setup(name='faceDetection',
      version=__version__,
      description='Multi-task Cascaded Convolutional Neural Networks for Face Detection, based on TensorFlow',
      long_description=readme(),
      packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
      install_requires=[
      ],
      classifiers=[
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      keywords="faceDetection face detection tensorflow pip package",
      zip_safe=False)
