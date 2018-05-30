#!/usr/bin/env python

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# for your packages to be recognized by python
d = generate_distutils_setup(
 packages=['utils','face_classification_ros'],
 package_dir={'utils':'src/utils','face_classification_ros': 'ros/src/face_classification_ros'}
)

setup(**d)
