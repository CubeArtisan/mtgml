import sys

from skbuild import setup
from setuptools import find_packages

setup(
    name="mtgml-native-generators",
    version="0.0.1",
    description="A faster more scalable version of the pure python keras Sequence.",
    license="AGPL",
    packages=find_packages(where='mtgml'),
    package_dir={"": "mtgml"},
    cmake_install_dir="mtgml/generated",
    include_package_data=True,
)
