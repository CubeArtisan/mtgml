import sys

from setuptools import find_packages
from skbuild import setup

setup(
    name="mtgml-native-generators",
    author="Devon Richards(ruler501)",
    author_email="support@cubeartisan.net",
    version="0.0.2",
    description="A faster threaded generators for loading mtgml training data.",
    # license="AGPL",
    packages=["mtgml/" + x for x in find_packages(where="mtgml")],
    cmake_install_dir="mtgml_native/generators",
    include_package_data=True,
    # python_requires=">=3.8",
    zip_safe=False,
)
