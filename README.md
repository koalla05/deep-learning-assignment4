# deep-learning-assignment4

Transformer encoders for the KSE location extraction competition: find the places mentioned in Ukrainian
and Russian Telegram posts. It's the same task as in Assignment 3.

## EDA and metric analysis

Most of the EDA is in the Assignment 3 report, so here we only repeat what matters for this part. The new
numbers are in [`eda.ipynb`](eda.ipynb).

We train on short labeled sentences (about 1M Ukrainian and 8M Russian) and get scored on 477 Telegram
posts. The posts are a different kind of text. They are about 3.5 times longer (395 characters on average
against 102-119), most of them have line breaks and emoji, and they mention more places. The 26 labeled
example posts also follow a different labeling style: 21.9% of their places include a word like "вулиця"
or "проспект", against 2-4% in the training data.

XLM-RoBERTa reads text as tokens, pieces of words from its vocabulary, and takes at most 512 of them. So we
counted tokens:

| Texts | Count | Median | 95th percentile | Max | Over 512 |
| --- | ---: | ---: | ---: | ---: | ---: |
| training sentences, uk | 31,432 | 27 | 54 | 85 | 0 |
| training sentences, ru | 31,956 | 30 | 59 | 217 | 0 |
| test posts | 477 | 73 | 314 | 926 | 8 |

128 tokens are enough for training. The 8 longest posts don't fit, so at prediction time we have to split
long posts into overlapping chunks.

### Metric

For every post Kaggle compares the set of predicted places with the true set, and counts matches, extra
places and missed places over the whole test set. F1 combines precision (how many of our predictions are
right) and recall (how many of the true places we found). Before comparing, the scorer lowercases the
strings and drops digits and one-letter words, so "м. Київ" matches "Київ". Our copy is in
[`evaluation/metric.py`](evaluation/metric.py).

The metric fits the task because it scores the final list of places, so any model and any postprocessing
can be compared with it. The problem is how strict it is. If the label is "Гоголівська" and we predict
"вулиці Гоголівська", we get one extra and one missed place, with nothing for the shared word. "Києві" and
"Київ" are different places. Posts with many places weigh more than short ones, and a real place that the
annotator forgot counts against us.

The biggest edge case is punctuation. The cleaning is supposed to remove it, but the regex on the
evaluation page is missing an `f` before the string, so commas, `!`, `?` and dashes stay. We tested it:
our best Assignment 3 file scored 0.59475, and the same file with `!` added to every name scored 0.00000.
Another one: a post with no places gives no credit for predicting nothing, only a penalty for predicting
something.

Besides F1 we look at precision and recall separately, F1 per language (90% of the posts are Ukrainian),
and recall on names that never appear in training. That was our weakest point in Assignment 3: the first
models found 76-87% of the names they had seen and about a quarter of the new ones. We also score the 26
labeled posts, but they are too few to pick a model with.

As an alternative metric we would use partial-match F1, where a prediction gets credit for the words it
shares with the true place. If the label is "проспекті Івасюка" and we predict "Івасюка", the pair gets
0.67 (word-level F1 with precision 1 and recall 0.5) instead of 0. The test labels often include words
like "проспект" that our training labels don't, so many of our errors are border errors like this one,
and the official metric treats them as completely wrong. The leaderboard still ranks by the official
metric, so that's the one we use to choose models.

## Validation

We use the same 3 folds as in Assignment 3 ([`data/cv_rows.parquet`](data/README.md), 63,388 sentences).
In cross-validation (CV) the model for fold k trains on the other two folds and is tested on fold k, and
the CV score is the mean of the three. Copies of one sentence, and Russian sentences from one document,
always stay in one fold, otherwise the model could just remember them. With the same folds for every model,
RNN and Transformer scores can be compared directly, and an ensemble gets an honest CV score too. There is
no time split because the data has no dates. A validation set of real posts would be better, but we only
have 26 labeled ones, and labeling test posts by hand is against the rules.

For every experiment we write down the CV score, the F1 on the 26 labeled posts and the public leaderboard
score. We tune only on CV, never on the 26 posts or the leaderboard. Before you submit, strip punctuation
from both ends of every predicted place (see the metric section).

### How well CV follows the leaderboard

In Assignment 3 it didn't. We took the 14 submissions that had both scores and computed the Spearman rank
correlation: 1 means CV and the leaderboard put the models in the same order, 0 means no relation. It came
out 0.20, and -0.28 for the 12 CRF models without the two weak pilots. Neither number is significant. The
BiLSTM was second on CV and last on Kaggle. The full table is in [`eda.ipynb`](eda.ipynb).

We see a few reasons. CV tests on sentences and the leaderboard on posts, which look very different (see
below). The test labels follow other rules, and since the metric only accepts exact strings, a border or
case difference costs the whole place. The CV scores of the 12 CRF models are within 0.019 of each other,
close to the 0.013 we got just from changing the random seed, so most of their order is noise. And the
public leaderboard uses only part of the 477 posts, so a few posts can move it. The two scores agree only
when a change helps on both kinds of text. The character encoder in Assignment 3 improved recall on new
names, and it raised both CV (0.891 to 0.896) and the leaderboard (0.589 to 0.595).

### Adversarial validation

To check how different the posts are, we trained a classifier to tell training sentences from test posts
and measured ROC AUC on texts it didn't see. AUC near 0.5 means it can't tell them apart, 1.0 means it
always can. As the assignment asks, the classifier is the same model as the main task, `xlm-roberta-base`.
The setup is in [`adversarial_validation.ipynb`](adversarial_validation.ipynb).

| Run | Held-out texts | AUC |
| --- | ---: | ---: |
| raw text | 287 | 0.992 |
| no links, emoji or line breaks, every text cut to 32 tokens | 150 | 0.972 |

The second run hides format and length, and the model still gets 0.97, so the difference is in the content
itself. The Assignment 3 BiLSTM got 0.94 and 0.84 in the same two settings. A good CV score on sentences
doesn't promise a good score on posts, and that's why we report the leaderboard score next to CV for every
experiment.
