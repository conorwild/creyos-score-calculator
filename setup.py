from setuptools import setup, find_packages

setup(
    name="composite_scores",
    version="0.0.2",
    author="Conor J. Wild",
    author_email="cwild@uwo.com",
    description="Composite Score Calculator Based on Wild et al. 2022 (COVID Cognition)",
    packages=find_packages(),
    package_data={
        "composite_scores": ["models/*"],
    },
    extras_require={
        "dev": [
            "datalad",
            "ipykernel",
            "matplotlib",
            "plotly",
        ]
    },
    install_requires=[
        "scikit-learn~=1.5.2",
        "numpy > 2.0.0",
        "factor-analyzer",
        "pandas~=2.2.0",
        "statsmodels",
        "joblib~=1.2.0",
    ],
)
