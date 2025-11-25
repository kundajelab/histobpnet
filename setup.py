from setuptools import setup, find_packages

setup(
    name="histobpnet",
    version="0.0.1",
    packages=find_packages(),
    include_package_data=True,
        install_requires=[
        "lightning",
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "pyfaidx",
        "pyBigWig",
        "pybedtools",
        "intervaltree",
        "tqdm",
        "wandb",
        "PyYAML",
        "accelerate",
        "matplotlib",
        "enformer_pytorch",
        "tangermeme",
        "polars",
        "pooch",
        "toolbox @ git+https://github.com/watiss/toolbox.git@main"
    ],
)