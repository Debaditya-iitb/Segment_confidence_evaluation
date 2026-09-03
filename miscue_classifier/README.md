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
53-column `segment_features.py` output they are columns 52–54. 

## Quick start

```bash


# 1. produce features (from the parent folder)
python ../segment_features.py --lang hi --csv segments.csv --out features.csv

# 2. classify
python predict.py --input features.csv --output scored.csv


python predict.py --input .. /demo_output_hindi.csv --output /demo/demo_hindi_scored.csv --slim


```

Example of the slim output:

| utterance_id | canonical_word | start_time | end_time | costed_neglog | entropy_gibbs_exp_min | miscue_prob | miscue_label | miscue_pred |
|---|---|---|---|---|---|---|---|---|
| …HI-KV-001_1.wav | एक | 2.34 | 2.67 | 0.0 | 0.0027 | 0.1148 | 0 | C |
| …HI-KV-001_1.wav | राजा | 2.67 | 3.15 | 0.0 | 0.0048 | 0.0633 | 0 | C |



## Models

Pick by (a) which timestamps you fed `segment_features.py`, and (b) how many features you have.

| Model | Features | AUC | AP | Miss @5% FPR | F1 | Precision | Recall |
|---|---|---|---|---|---|---|---|
| `miscue_clf_Cano_decoded_ed_conf2.joblib` *(default)* | costed_neglog, entropy_gibbs_exp_min | 0.8893 | 0.4902 | **36.92%** | 0.5301 | 0.4571 | 0.6308 |
| `miscue_clf_MT_decoded_ed_conf2.joblib` | costed_neglog, entropy_gibbs_exp_min | **0.8961** | 0.4738 | 38.90% | 0.5220 | 0.4557 | 0.6110 |
| `miscue_clf_Cano_decoded_ed1.joblib` | costed_neglog | 0.8130 | 0.4047 | 38.08% | 0.5228 | 0.4525 | 0.6192 |
| `miscue_clf_MT_decoded_ed1.joblib` | costed_neglog | 0.8200 | 0.3982 | 42.34% | 0.5000 | 0.4413 | 0.5766 |





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


## Input

`predict.py` needs only the feature columns, both emitted by `segment_features.py`:

- `costed_neglog` — costed edit distance between canonical and decoded phones
- `entropy_gibbs_exp_min` — frame-confidence feature (for the `ed_conf2` models)

Everything else in the file is passed through untouched. Missing values are fine — the tree
handles NaN natively (`costed_neglog` is NaN where there was no decoded string).


## Threshold

`miscue_label = miscue_prob >= threshold`. The shipped threshold is the point giving a **5%
false-positive rate** on KV-1908 (0.8036 for the default model). Raise it for fewer
false alarms and more misses; lower it for the reverse.

