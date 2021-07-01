#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import os

import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

# Same version number in code and pip
# https://packaging.python.org/guides/single-sourcing-package-version/
# Option 3
version = {}
with open(os.path.join("dstain", "__version__.py")) as f:
    exec(f.read(), version)

setuptools.setup(
    name="dstain",
    description="Digital IHC staining from H&E slides",
    long_description=long_description,
    author="Bryan He",
    author_email="bryanhe@stanford.edu",
    version=version["__version__"],
    url="https://github.com/bryanhe/digital-staining",
    packages=setuptools.find_packages(),
    install_requires=[
        "captum",
        "click",
        "flask",
        "matplotlib",
        "numpy",
        "opencv-python",
        "openslide-python",
        "scikit-image",
        "scipy",
        "sklearn",
        "tifffile",
        "torch",
        "torchvision",
        "tqdm",
        "umap-learn"
    ],
    tests_require=[
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    entry_points={
        "console_scripts": [
            "dstain=dstain:main",
            "deepzoom=dstain.command:deepzoom",
            "register=dstain.command:register",
            # TODO: new entry point for running on new image?
        ],
    }
)
