# %%
import cbspython as cbs
from factor_analyzer import FactorAnalyzer
from IPython.display import display
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from composite_scores import (
    CompositeScores,
    DomainScores,
    ProcessingSpeed,
    load_CC_norms,
)

# Display options for in this notebook
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")  # pyright: ignore
np.set_printoptions(precision=3)

# %%
Ynorm = load_CC_norms()
af_ = cbs.all_cbs_columns(Ynorm)
df_ = DomainScores.required_features()
tf_ = ProcessingSpeed.required_features()
# %%
Ytfm = Pipeline(
    steps=[
        ("center", StandardScaler(with_mean=True, with_std=True)),
        ("yeo", PowerTransformer(method="yeo-johnson")),
    ]
).fit(Ynorm[af_].values)

Znorm = Ynorm.copy()
Znorm[af_] = Ytfm.transform(Znorm[af_].values)

pca_dmn_norm = FactorAnalyzer(method="principal", n_factors=3, rotation="varimax").fit(
    Znorm[df_]
)

domains = ["STM", "reasoning", "verbal"]
loadings_domain = pd.DataFrame(pca_dmn_norm.loadings_, index=df_, columns=domains)

print("Normative Data Loadings:")
display(loadings_domain)

# Calculates the 3 cognitive domain scores from the fitted PCA model
Znorm[domains] = pca_dmn_norm.transform(Znorm[df_])

# Measure of processing speed: take the 1st Principal Component across
# timing-related features (the list of tf_), derived from ctrl group data.
pca_spd_norm = FactorAnalyzer(
    method="principal", n_factors=1, rotation=None  # pyright: ignore
).fit(Znorm[tf_])

# Force it to go the right away around, so higher scores are better
if pca_spd_norm.loadings_.mean() > 0:  # pyright: ignore
    pca_spd_norm.loadings_ *= -1  # pyright: ignore

loadings_speed = pd.DataFrame(
    pca_spd_norm.loadings_, index=tf_, columns=["processing_speed_01"]
)

Znorm["processing_speed_01"] = pca_spd_norm.transform(Znorm[tf_])
display(loadings_speed)

# Overall measure across CBS battery: the average of all 12 task z-scores,
# then rescale to have M = 0.0, SD = 1.0
Znorm["overall"] = Znorm[df_].mean(axis=1)
overall_tfm = StandardScaler(with_mean=True, with_std=True).fit(Znorm[["overall"]])
Znorm["overall"] = overall_tfm.transform(Znorm[["overall"]])

comp_scores = cbs.DOMAIN_NAMES + ["processing_speed_01", "overall"]

# %%
print("Composite Score Loadings:")
comp_score_calc = CompositeScores.load_pretrained()
display(comp_score_calc.domains.loadings)
display(comp_score_calc.processing_speed.loadings)
# %%
from pandas.testing import assert_frame_equal

assert_frame_equal(
    Znorm[comp_scores], comp_score_calc.transform(Ynorm[df_ + tf_])[comp_scores]
)

# %%
