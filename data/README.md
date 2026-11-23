# Data

`cv_rows.parquet` has the 63,388 training sentences we used for cross-validation in Assignment 3, with
their fold (21,129 / 21,130 / 21,129 rows). It is the `cv` part of `model/cache/rows.parquet` from the
Assignment 3 repo.

Columns you will need:

- `text`: the sentence
- `loc_markers`: character spans of the locations, same format as in the Kaggle files
- `fold`: 0, 1 or 2
- `source`: `uk` or `ru`
- `group_id`: a group never ends up in two folds. For Ukrainian data a group is one text after
  normalization, for Russian data it is a set of documents that share sentences.

`test.csv` (477 posts for the submission) and `labeling_sample.csv` are copied from Kaggle as they are.

The full `uk_geo_dataset.csv` (211 MB) and `ru_geo_dataset.csv` (2 GB) are too big for GitHub. If you
need them, download them from the competition page.
