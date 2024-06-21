# %%
from IPython.display import display
import pandas as pd

from composite_scores import (
    CompositeScores,
    DomainScores,
    ProcessingSpeed,
    load_CC_norms,
)

pd.set_option("display.float_format", lambda x: f"{x:.4f}")  # pyright: ignore

# %%
Ynorm = load_CC_norms()
df_ = DomainScores.required_features()
tf_ = ProcessingSpeed.required_features()

comp_score_calc = CompositeScores(overall_features=df_).fit(Ynorm[df_ + tf_])
comp_score_calc.domains.names = ["STM", "Reasoning", "Verbal"]
display(comp_score_calc.domains.loadings)  # pyright: ignore
comp_score_calc.save_pretrained()

# %%
