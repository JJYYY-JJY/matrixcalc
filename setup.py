#!/usr/bin/env python3
"""
Setup script for the Gaussian Elimination Solver package.

This allows the package to be installed in development mode using:
    pip install -e .

This makes the package importable from anywhere and solves pytest import issues.
"""

from setuptools import setup, find_packages

# Read the README for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gaussian-elimination-solver",
    version="1.0.0",
    author="Gaussian Elimination Solver Team",
    author_email="",
    description="A robust implementation of Gaussian elimination with partial pivoting for solving linear systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gaussian-elimination-solver",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=6.0.0"],
        "test": ["pytest>=6.0.0"],
    },
    entry_points={
        "console_scripts": [
            "gaussian-demo=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 