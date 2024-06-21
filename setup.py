from setuptools import setup, find_packages

setup(
    name="composite_scores",
    version="0.0.1",
    author="Conor J. Wild",
    author_email="conor.wild@creyos.com",
    description="Composite Score Calculator Based on Wild et al. 2022 (COVID Cognition)",
    packages=find_packages(),
    package_data={
        "composite_scores": ["models"],
    },
    extras_require={
        "dev": [
            "datalad",
            "creyon@git+https://bitbucket.org/cambridgebrainsciences/creyon.git@main",  # noqa: E501,
            "ipykernel",
            "matplotlib",
            "plotly",
        ]
    },
    install_requires=[
        "scikit-learn~=1.5.0",
        "factor-analyzer",
        "pandas~=2.2.0",
        "scipy~=1.10.0",
        "numpy~=1.24.0",
        "statsmodels~=0.13.0",
        "joblib~=1.2.0",
    ],
)
