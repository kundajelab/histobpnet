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
        # "yaml",
        "PyYAML",
        "accelerate",
        "matplotlib",
        "enformer_pytorch",
        "tangermeme",
        # "session-info", # for debug logging (referenced from the issue template)
        "polars",
    ], # Add dependencies here
)

