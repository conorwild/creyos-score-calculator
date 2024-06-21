import cbspython as cbs
import pandas as pd

from composite_scores.data.covidcog.cbs_data.normative_data import (
    NormativeData as Norms,
)
from composite_scores.data.covidcog.covid_cognition.lib_utils import (
    report_N,
    remove_unused_categories,
    set_column_names,
)


def load_CC_norms() -> pd.DataFrame:
    Ynorm = Norms.score_data
    af_ = cbs.abbrev_features(list(Ynorm.columns))

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

    Ynorm = (
        Ynorm.pipe(set_column_names, af_)  # pyright: ignore
        .reset_index("device_type")
        .pipe(report_N, "initial dataset", reset_count=True)
        .query('~(device_type in ["BOT", "CONSOLE", "MOBILE"])')
        .pipe(report_N, "drop unsupported devices")
        .reset_index()
        .astype({"user": str})
        .set_index("user")
    )

    Qnorm = (
        Norms.questionnaire.data.reset_index()
        .astype({"user": str})
        .set_index("user")
        .rename(columns={"SES_growing_up": "SES", "sex": "gender"})
        .assign(post_secondary=lambda x: x.education >= "Bachelor's Degree")
        .assign(nicotine=lambda x: x.cigarettes_per_day > 0)
        .assign(alcohol=lambda x: x.alcohol_per_week > 14)
        .assign(exercise=lambda x: x.exercise_freq >= "Once or twice a week")
        .pipe(report_N, "initial dataset", reset_count=True)  # pyright: ignore
    )

    return (
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
