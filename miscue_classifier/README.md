# Miscue classifier

Takes the CSV that `segment_features.py` produces and decides, per word, whether the child
read it **correctly (C, label 0)** or produced a **miscue (S, label 1)**.

```
segment_features.py  ──▶  features.csv  ──▶  predict.py  ──▶  scored.csv
                                                              + miscue_prob   (probability)
                                                              + miscue_label  (1 = S, 0 = C)
                                                              + miscue_pred   ("S" / "C")
```

The three prediction columns are **appended at the end** of the input CSV, so in a
53-column `segment_features.py` output they are columns 52–54. Pass `--slim` to also get
a compact copy with only the identifying and prediction columns.

## Quick start

```bash
conda activate deb_wav2vec          # see Environment below

# 1. produce features (from the parent folder)
python ../segment_features.py --lang hi --csv segments.csv --out features.csv

# 2. classify
python predict.py --input features.csv --output scored.csv

# compact copy alongside, easier to eyeball
python predict.py --input features.csv --output scored.csv --slim
```

Example of the slim output:

| utterance_id | canonical_word | start_time | end_time | costed_neglog | entropy_gibbs_exp_min | miscue_prob | miscue_label | miscue_pred |
|---|---|---|---|---|---|---|---|---|
| …HI-KV-001_1.wav | एक | 2.34 | 2.67 | 0.0 | 0.0027 | 0.1148 | 0 | C |
| …HI-KV-001_1.wav | राजा | 2.67 | 3.15 | 0.0 | 0.0048 | 0.0633 | 0 | C |

Other commands:

```bash
python predict.py --list-models                 # what is available
python predict.py --info                        # model card for the default model
python predict.py --input features.csv --model models/miscue_clf_MT_decoded_ed1.joblib
python predict.py --input features.csv --threshold 0.65   # trade recall vs false alarms
```

## Models

Pick by (a) which timestamps you fed `segment_features.py`, and (b) how many features you have.

| Model | Features | AUC | AP | Miss @5% FPR | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| `miscue_clf_Cano_decoded_ed_conf2.joblib` *(default)* | costed_neglog, entropy_gibbs_exp_min | 0.8893 | 0.4902 | **36.92%** | 0.5301 | 0.4571 | 0.6308 |
| `miscue_clf_MT_decoded_ed_conf2.joblib` | costed_neglog, entropy_gibbs_exp_min | **0.8961** | 0.4738 | 38.90% | 0.5220 | 0.4557 | 0.6110 |
| `miscue_clf_Cano_decoded_ed1.joblib` | costed_neglog | 0.8130 | 0.4047 | 38.08% | 0.5228 | 0.4525 | 0.6192 |
| `miscue_clf_MT_decoded_ed1.joblib` | costed_neglog | 0.8200 | 0.3982 | 42.34% | 0.5000 | 0.4413 | 0.5766 |

Metrics are pooled out-of-fold from 5-fold CV on KV-1908 (`fold` column), evaluated on rows
whose ASR verdict is `c` or `s`. The shipped model is refitted on all such rows
(123,486 Cano / 123,764 MT).

**Adding the confidence feature is worth ~0.08 AUC** and 1–3 points of miss rate. Use
`ed_conf2` unless you only have the edit distance.

## Hyperparameters

`sklearn.ensemble.HistGradientBoostingClassifier`:

| Parameter | Value | Why |
|---|---|---|
| `learning_rate` | 0.06 | project's production setting |
| `max_leaf_nodes` | 15 | heavy regularisation; 31/63 tested, no gain |
| `min_samples_leaf` | 50 | wins almost universally in tuning |
| `l2_regularization` | 1.0 | |
| `class_weight` | `'balanced'` | inert at a fixed-FPR operating point, kept for safety |
| `early_stopping` | `True` | 40/15-iteration caps were measurably under-trained |
| `max_iter` | 500 | a ceiling; early stopping decides |
| `random_state` | 42 | |

A nested-CV sweep over `learning_rate` ∈ {0.03, 0.06, 0.12} × `max_leaf_nodes` ∈ {15, 31, 63}
× `min_samples_leaf` ∈ {50, 200} changed AUC by at most **0.0012** and miss rate by at most
**0.13 points**, with the inner loop agreeing on a configuration only 20–60% of the time.
The grid points are effectively equivalent, so these values are fixed rather than tuned.

A tree is used rather than logistic regression deliberately: on this data a tree beats
`BalancedBagging(LogisticRegression)` by 0.007–0.027 AUC and up to 19 points of miss rate.

## Input

`predict.py` needs only the feature columns, both emitted by `segment_features.py`:

- `costed_neglog` — costed edit distance between canonical and decoded phones
- `entropy_gibbs_exp_min` — frame-confidence feature (for the `ed_conf2` models)

Everything else in the file is passed through untouched. Missing values are fine — the tree
handles NaN natively (`costed_neglog` is NaN where there was no decoded string to diff).

**Do not substitute a `costed_neglog` column from elsewhere.** These models were trained on
distances computed with `matrices/confusion_hindi_expt.npz` (key `cost_neglog`) — the same
matrix `segment_features.py` uses. The KV-1908 master file's `*_costed_neglog` columns come
from different matrices and are **not** on the same scale; mixing them changed 7% of labels
in a test.

## Threshold

`miscue_label = miscue_prob >= threshold`. The shipped threshold is the point giving a **5%
false-positive rate** on held-out KV-1908 (0.8036 for the default model). Raise it for fewer
false alarms and more misses; lower it for the reverse. At 5% FPR these models miss roughly
37–42% of genuine miscues — that is the honest operating point, not a rounding error.

## Environment

Models are pickled under scikit-learn 1.7.2 / numpy 2.2.6. Loading under a newer stack fails
with `PCG64 is not a known BitGenerator module`. Either use `conda activate deb_wav2vec` on
this cluster, or `pip install -r requirements.txt`. To move off this stack entirely, re-fit
with `_train.py` in the target environment.

## Files

| Path | What |
|---|---|
| `predict.py` | CLI — the only thing you need to run |
| `models/*.joblib` | four trained classifiers, each bundling its threshold and model card |
| `manifest.json` | model index with metrics |
| `_train.py` | training script, for reproducing or re-fitting |
| `run_train_slurm.sh` | SLURM wrapper for `_train.py` |
| `demo/demo_hindi_scored.csv` | worked example — `../demo/demo_output_hindi.csv` scored |
| `requirements.txt` | pinned versions |

## Caveats

- **Population.** Trained and evaluated only on words whose ASR verdict is `c` or `s`. In the
  full Stage-2 cascade, ASR insertions and deletions are flagged as miscues *before* the
  classifier runs, so end-to-end system numbers are better than the table above.
- **The bundled cost matrix is experiment-scoped.** `confusion_hindi_expt.npz` was fit on all
  available pairs and used to score rows from that same pool, with no fold split. It is
  mildly optimistic. For a clean number, rebuild it per fold.
- **Lexicon gap.** 2.81% of KV-1908 rows have `<UNK>` canonical phones from eight
  out-of-lexicon words (कहानियां, भ्रमन, पडा, कंकड, आंखें, पांच, मां, संपूर्न — mostly
  anusvāra/chandrabindu and न/ण spelling variants). Their edit distance is inflated ~17×.
  Fixing the lexicon will change these numbers.
- **Hindi only.** An English model would need re-training with `confusion_english_expt.npz`.
