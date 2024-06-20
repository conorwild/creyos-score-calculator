# %%
from joblib import dump
from typing import cast

import cbspython as cbs
from factor_analyzer import FactorAnalyzer
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler

from composite_scores import (
    CompositeScores,
    DomainScores,
    OverallScore,
    ProcessingSpeed,
)
from composite_scores.data.covidcog.cbs_data.normative_data import (
    NormativeData as Norms,
)
from composite_scores.data.covidcog.covid_cognition.lib_utils import (
    report_N,
    remove_unused_categories,
    set_column_names,
)

# Display options for in this notebook
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: "%.4f" % x)
np.set_printoptions(precision=3)

# %%#%% [markdown]
# ## Normative (Pre-Pandemic) Dataset
# ### Load & Preprocess Data
# %%
# Loads the normative dataset from the (private) SS library
Ynorm = cast(pd.DataFrame, Norms.score_data)

# List columns corresponding to "timing" (RT) features
tf = cbs.timing_features(exclude=["spatial_planning"])  # SP does not have one
tf_ = cbs.abbrev_features(tf)  # Abbreviated name

# List of columns corresponding to score features used in domain score calcs.
df = cbs.domain_feature_list()
df_ = cbs.abbrev_features(df)

# A list of "all" available score features
af = list(Ynorm.columns)
af_ = cbs.abbrev_features(af)

# From Hampshire et al. 2012, and Wild et al. These are the variables that are
# related to performance on these cognitive tasks.
Xcovar = [
    "age",
    "gender",
    "post_secondary",
    "SES",
    "exercise",
    "nicotine",
    "alcohol",
    "cannabis",
    "stimulants",
    "depressants",
]

# Loads the norm dataset (Sleep Study, 2017)
print("Normative Data - Scores:")
Ynorm = (
    Ynorm.pipe(set_column_names, af_)
    .reset_index("device_type")
    .pipe(report_N, "initial dataset", reset_count=True)
    .query('~(device_type in ["BOT", "CONSOLE", "MOBILE"])')
    .pipe(report_N, "drop unsupported devices")
    .reset_index()
    .astype({"user": str})
    .set_index("user")
)

# Loads and organises the Norms questionnaire dataset
# Have to rename a few columns to match them up to the new study data
print("\nNormative Data - Questionnaires:")
Qnorm = (
    Norms.questionnaire.data.reset_index()
    .astype({"user": str})
    .set_index("user")
    .rename(columns={"SES_growing_up": "SES", "sex": "gender"})
    .assign(post_secondary=lambda x: x.education >= "Bachelor's Degree")
    .assign(nicotine=lambda x: x.cigarettes_per_day > 0)
    .assign(alcohol=lambda x: x.alcohol_per_week > 14)
    .assign(exercise=lambda x: x.exercise_freq >= "Once or twice a week")
    .pipe(report_N, "initial dataset", reset_count=True)
)

# Join the test scores (Ynorm) and the questionnaire data (Qnorm), then
# filter score columns to remove outliers (6 then 4 stdevs)
print("\nNormative Dataset:")
Znorm = (
    Ynorm.join(Qnorm[Xcovar], how="inner")
    .pipe(report_N, "join datasets", reset_count=True)
    .query("(age >= 18) & (age <= 100)")
    .query('gender in ["Male", "Female"]')
    .pipe(report_N, "filter age")
    .pipe(report_N, "English only")
    .dropna(subset=Xcovar + af_)
    .pipe(report_N, "drop missing data")
    .pipe(cbs.filter_by_sds, subset=af_, sds=[6], drop=True)
    .pipe(report_N, "6 SD filter")
    .pipe(cbs.filter_by_sds, subset=af_, sds=[4], drop=True)
    .pipe(report_N, "4 SD filter")
    .dropna()
    .pipe(remove_unused_categories)
    .pipe(report_N, "final")
)

Zorig = Znorm.copy()


# %%
Ytfm = Pipeline(
    steps=[
        ("center", StandardScaler(with_mean=True, with_std=True)),
        ("yeo", PowerTransformer(method="yeo-johnson")),
    ]
).fit(Znorm[af_].values)

Znorm[af_] = Ytfm.transform(Znorm[af_].values)

pca_dmn_norm = FactorAnalyzer(method="principal", n_factors=3, rotation="varimax").fit(
    Znorm[df_]
)

domains = ["STM", "reasoning", "verbal"]
loadings_norm = pd.DataFrame(pca_dmn_norm.loadings_, index=df_, columns=domains)
loadings_norm

# %%
# Calculates the 3 cognitive domain scores from the fitted PCA model
Znorm[domains] = pca_dmn_norm.transform(Znorm[df_])

# Measure of processing speed: take the 1st Principal Component across
# timing-related features (the list of tf_), derived from ctrl group data.
pca_spd_norm = FactorAnalyzer(method="principal", n_factors=1, rotation=None).fit(
    Znorm[tf_]
)

# Force it to go the right away around, so higher scores are better
if pca_spd_norm.loadings_.mean() > 0:
    pca_spd_norm.loadings_ *= -1

Znorm["processing_speed"] = pca_spd_norm.transform(Znorm[tf_])

# Overall measure across CBS battery: the average of all 12 task z-scores,
# then rescale to have M = 0.0, SD = 1.0
Znorm["overall"] = Znorm[df_].mean(axis=1)
overall_tfm = StandardScaler(with_mean=True, with_std=True).fit(Znorm[["overall"]])
Znorm["overall"] = overall_tfm.transform(Znorm[["overall"]])

comp_scores = cbs.DOMAIN_NAMES + ["processing_speed", "overall"]
test_scores = df_

Yvar = test_scores + comp_scores
Znorm[comp_scores]
# %%
comp_score_calc = CompositeScores(overall_features=df_).fit(Zorig[af_])
comp_score_calc.domains.loadings

# %%
comp_score_calc.domains.names = domains
comp_score_calc.transform(Zorig[af_])
# %%
