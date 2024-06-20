from typing import Any, get_args, List, Literal, Optional, Union

from factor_analyzer import FactorAnalyzer
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

DomainFeatures = Literal[
    "SS_max_score",
    "GR_final_score",
    "DT_final_score",
    "OOO_max",
    "ML_max_score",
    "RT_final_score",
    "FM_final_score",
    "DS_max_score",
    "SP_final_score",
    "PA_max_score",
    "PO_final_score",
    "TS_max_score",
]

TimingFeatures = Literal[
    "SS_avg_ms_per_item",
    "GR_avg_ms_correct",
    "DT_avg_ms_correct",
    "OOO_avg_ms_correct",
    "ML_avg_ms_per_item",
    "RT_avg_ms_correct",
    "FM_avg_ms_correct",
    "DS_avg_ms_per_item",
    "PA_avg_ms_per_item",
    "PO_avg_ms_correct",
    "TS_avg_ms_per_item",
]


class MyFactorAnalyzer(FactorAnalyzer):
    def __init__(self, _name_prefix: Optional[str] = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.output = "default"
        self._name_prefix = _name_prefix if _name_prefix is not None else "F"
        self.names = [f"{self._name_prefix}_{i+1:02d}" for i in range(self.n_factors)]

    def set_output(
        self, *, transform: Union[Literal["default", "pandas"], None] = None
    ):
        self.output = transform
        return self

    @property
    def names(self) -> List[str]:
        return self._names

    @names.setter
    def names(self, new_values: List[str]) -> None:
        assert len(new_values) == self.n_factors
        self._names = new_values

    def get_feature_names_out(self, features: Optional[List[str]] = None):
        """Not sure why we need / use the features param...?"""
        return self.names

    def transform(self, X: Union[pd.DataFrame, ArrayLike]) -> ArrayLike:
        if isinstance(X, pd.DataFrame) & (self.output == "pandas"):
            ix = X.index
            cols = pd.Index(self.get_feature_names_out())
            return pd.DataFrame(super().transform(X), index=ix, columns=cols)
        else:
            return super().transform(X)


class ColumnSelector(ColumnTransformer):
    def __init__(self, columns: List[Union[DomainFeatures, TimingFeatures]]):
        self.columns = columns
        super().__init__(
            [("selector", "passthrough", self.columns)],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def set_params(self, columns: List[str], **kwargs: Any):
        # Assumes that it is a one step transformer with only passthrough.
        passthrough = self.transformers[0]
        self.transformers[0] = (*passthrough[0:2], columns)
        super().set_params(**kwargs)


class DomainScores(Pipeline):
    @staticmethod
    def required_features() -> List[DomainFeatures]:
        return list(get_args(DomainFeatures))

    @property
    def n_factors(self) -> int:
        return 3

    @property
    def pca(self) -> MyFactorAnalyzer:
        return self.steps[-1][1]

    @property
    def loadings(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.pca.loadings_, index=self.required_features(), columns=self.names
        )

    @property
    def names(self) -> List[str]:
        return self.pca.names

    @names.setter
    def names(self, new_values: List[str]) -> None:
        self.pca.names = new_values

    def __init__(self):
        self._name_prefix = "domain"
        super().__init__(
            [
                ("selector", ColumnSelector(self.required_features())),
                (
                    "PCA",
                    MyFactorAnalyzer(
                        method="principal",
                        n_factors=self.n_factors,
                        rotation="varimax",
                        _name_prefix=self._name_prefix,
                    ),
                ),
            ]
        )


def calculate_average(X: pd.DataFrame) -> pd.Series:
    return X.mean(axis=1)


def overall_feature_name_out(
    self: FunctionTransformer, feature_names_in: List[str]
) -> List[str]:
    return ["overall"]


class OverallScore(Pipeline):

    @property
    def features(self) -> List[DomainFeatures]:
        return self._features

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, new_value: str) -> None:
        self._name = new_value

    @features.setter
    def features(self, new_features: List[DomainFeatures]) -> None:
        self._features = new_features

    def __init__(self, features: List[DomainFeatures]):
        self.features = features
        self.name = "overall"
        super().__init__(
            [
                ("selector", ColumnSelector(self.features)),
                (
                    "average",
                    FunctionTransformer(
                        calculate_average, feature_names_out=overall_feature_name_out
                    ),
                ),
                ("rescale", StandardScaler(with_mean=True, with_std=True)),
            ]
        )


class ProcessingSpeed(Pipeline):
    @property
    def names(self):
        return [f"{self._name_prefix}_{i+1:02d}" for i in range(self.n_components)]

    @staticmethod
    def required_features() -> List[TimingFeatures]:
        return list(get_args(TimingFeatures))

    def __init__(self, n_components: int = 1):
        self.n_components = n_components
        self._name_prefix = "processing_speed"
        super().__init__(
            [
                ("selector", ColumnSelector(self.required_features())),
                (
                    "PCA",
                    MyFactorAnalyzer(
                        method="principal",
                        n_factors=self.n_components,
                        rotation=None,
                        _name_prefix=self._name_prefix,
                    ),
                ),
            ]
        )

    def fit(self, X: Union[pd.DataFrame, ArrayLike], y: Optional[ArrayLike] = None):
        super().fit(X, y)
        pca_model = self.steps[-1][1]
        for n in range(self.n_components):
            if pca_model.loadings_[:, n].mean() > 0.1:
                pca_model.loadings_[:, n] *= -1
        return self


class CompositeScores(Pipeline):

    @property
    def domains(self) -> Pipeline:
        return self.scores.named_transformers["domains"]

    @property
    def overall(self) -> Pipeline:
        return self.scores.named_transformers["overall"]

    @property
    def processing_speed(self) -> Pipeline:
        return self.scores.named_transformers["processing_speed"]

    def __init__(self, overall_features: List[DomainFeatures]):

        self.overall_features = overall_features
        self.preproc = Pipeline(
            steps=[
                ("center", StandardScaler(with_mean=True, with_std=True)),
                ("yeo", PowerTransformer(method="yeo-johnson")),
            ]
        )

        self.scores = FeatureUnion(
            [
                ("domains", DomainScores()),
                ("overall", OverallScore(features=overall_features)),
                ("processing_speed", ProcessingSpeed()),
            ],
            verbose_feature_names_out=False,
        )

        super().__init__([("preproc", self.preproc), ("scores", self.scores)])
        self.set_output(transform="pandas")
