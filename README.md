# deep-learning-assignment4

Transformer encoders for the KSE location extraction competition.

## Validation setup

We keep the 3 folds from Assignment 3, so every model is compared on the same data. They are in
`data/cv_rows.parquet`.

For fold k, train on the other two folds and evaluate on fold k. The CV score of an experiment is the
mean over the 3 folds.

Score with `competition_f1` from `evaluation/metric.py`. It is the same code as on the Kaggle evaluation
page.

Strip punctuation from both ends of every predicted location before you submit. Kaggle's scorer doesn't
do it: our 0.59475 submission scored 0.00000 after we added `!` to every name.

Write down both the CV score and the public LB score for every experiment.
