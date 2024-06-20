# Composite Score Calculators

![Schematic Here](images/composite_score_schematic.png "Title")

# Installation
### For general use
`pip3 install

# Usage
```
from composite_scores import CompositeScores
score_calculator = CompositesScores.load_pretrained()
new_score_df = score_calculator.transform(original_data_df[score_calculator.feature_names_in_])
```
- Notice that you have to select only the columns from your original data that used by the score calculator pipeline.

# Developer Notes
- Don't forget to do a `datalad get composite_scores/data/covidcog` to pull the data and code for that sub-package.

# TODO
- Improve the package structure so this can be directly installed from the GIN server