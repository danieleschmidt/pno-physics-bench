"""Setup script for pno-physics-bench."""

from setuptools import setup, find_packages

setup(
    name="pno-physics-bench",
    version="0.1.0",
    description="Uncertainty benchmarking for Physics-informed Neural Operators on PDE problems",
    author="Daniel Schmidt",
    author_email="danschmidt88@gmail.com",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21",
        "scipy>=1.7",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
