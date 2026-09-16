# English decision-tree miscue classifier (KV G3/4/5 Baseline, Cano system)

A trained, saved two-feature decision tree that scores segment_features.py
output for reading miscues. Mirrors github_en_miscue_v2/ (its own
segment_features.py + predict.py + shipped .joblib model), with a decision
tree instead of a 2-feature logistic model, features chosen at train time via
--edit/--confidence instead of hardcoded.

The model is already trained and saved at
classification/models/miscue_tree_EN_cano.joblib. Nothing needs to be
retrained to use it: give predict.py any CSV shaped like segment_features.py
output and it returns miscue predictions. No fold column, no training data,
no GT labels required at inference time.

## Environment

Every step (GPU decode and CPU tree training/inference) runs in the w2v
conda env, so numpy/torch/transformers are the same versions the wav2vec2
model was trained with (numpy 1.26.4, torch 2.3.1+cu121, transformers 4.38.2).

```bash
source /home/daplab/30006664/spack/linux-nehalem/miniconda3-24.7.1-pfw3umki5a6bht3kflyum226yawke6x3/etc/profile.d/conda.sh
conda activate w2v
# or, to recreate w2v elsewhere:
pip install -r requirements.txt
```

## Layout

```
tree_classifier_EN/
  requirements.txt
  README.md
  segmentation/                self-contained copy of segment_features.py
    segment_features.py        (same code/task as github_en_miscue_v2/segment_features.py)
    matrices/confusion_english_KV_G345_new_foldavg.npz
    run_segfeat_slurm.sh       full-corpus GPU decode
    master_EN_cano_segfeat_v1.csv   (output: phones + 33 confidence features + costed_neglog)
  classification/
    train.py                   fits + saves a model (--edit/--confidence pick the features)
    predict.py                 loads a saved model, scores new data, no fold needed
    models/
      miscue_tree_EN_cano.joblib     <- the shipped, ready-to-use model
    hparam_search.py / .log / _results.csv    tree hyperparameter tuning (evidence)
    reference/                 superseded fold-report scripts, kept for history
      step2_editdist_cv.py
      tree_classifier_en.py
      ...
  demo/                        12-utterance walkthrough: segmentation -> predict.py
```

## How segmentation works

segmentation/segment_features.py is the same code and does the same job as
github_en_miscue_v2/segment_features.py (copied in so this folder is
self-contained): one wav2vec2 forward pass per utterance, sliced at each
word's Cano_start_sec/Cano_end_sec window, producing per row:

- EN_phone_seq_raw / EN_phone_seq_clean - the greedy CTC decode
- 33 confidence features (raw_logmax_min, temprature_entropy_sum, ...,
  11 measures x {min, sum, mean})
- costed_neglog / lev_dist - edit distance between the decode and
  canonical_phone_seq, using the cost matrix passed via --matrix

```bash
sbatch segmentation/run_segfeat_slurm.sh
```

runs it over the full labelled master (master_EN_KV_G345_phones_v2.csv) with
segmentation/matrices/confusion_english_KV_G345_new_foldavg.npz as the cost
matrix, writing segmentation/master_EN_cano_segfeat_v1.csv. To score your own
data, run segment_features.py yourself with the SAME --matrix file (see
--help for all flags); using a different matrix changes the costed_neglog
scale and the shipped model's threshold will no longer be meaningful.

## Training (already done - for retraining only)

```bash
cd classification
python3 train.py --input ../segmentation/master_EN_cano_segfeat_v1.csv \
    --edit costed_neglog --confidence auto
```

- --edit names the edit-distance column to use. Default/shipped model uses
  plain costed_neglog (segment_features.py's own column, from the
  fold-averaged matrix) so the model can score any segment_features.py output
  directly. Pass --edit costed_neglog_cv to instead have train.py rebuild a
  leave-one-fold-out costed_neglog itself (needs a fold column in the input,
  and produces a model whose edit feature new data would then have to
  replicate the same way - not deployable without a fold column, kept as an
  option for research use, see classification/reference/).
- --confidence names one of the 33 confidence columns, or auto (default) to
  CV-select the best one. train.py still needs a fold column for this
  auto-selection and for the CV metrics reported below - that's a train-time
  requirement only, not an inference-time one.
- Writes classification/models/<--model-name>.joblib (default
  miscue_tree_EN_cano.joblib): the fitted DecisionTreeClassifier, the two
  feature names, the override rule, the 5%-FPR threshold, hyperparameters,
  and the CV metrics that threshold was chosen from.

### Shipped model

Trained on the full 88,576-row corpus, features costed_neglog +
temprature_logmax_min (CV-selected), tree max_depth=15, min_samples_leaf=100,
class_weight="balanced" (tuned by hparam_search.py: 96 configs x 33
confidence features x 4 folds, selecting on miss rate @5% FPR with the
override applied).

| override | AUC | Miss rate @5% FPR | threshold |
|---|---|---|---|
| True (deployed - S/D rows forced to prob 1.0) | 0.8926 | 42.81% | 0.858850 |
| False (model only) | 0.8713 | 60.63% | - |

(The untuned max_depth=4, min_samples_leaf=200 tree, as in the Hindi
reference wav2vec_deb/Results_Redo_classification/tree_classifier.py, scored
notably worse; see classification/reference/ for that history and the
costed_neglog_cv variant's numbers.)

## Scoring new data - predict.py

```bash
python3 predict.py --input your_features.csv --output your_scored.csv --slim
```

your_features.csv is segment_features.py's output (any CSV with at least
costed_neglog and temprature_logmax_min columns - the two features baked
into the shipped model; run segmentation first if you only have raw audio +
timings). Appends four columns, the last four in the output:

| column | meaning |
|---|---|
| miscue_forced | 1 = decided by the override, not the tree |
| miscue_prob | probability of a miscue, 0-1 (NaN if a feature was missing and the row wasn't overridden) |
| miscue_label | 1 = miscue, 0 = correct, -1 = unscorable (missing feature, not caught by the override) |
| miscue_pred | S / C / U for the three cases above |

Override: rows already labelled s or d by your ASR/aligner (read from
--asr-label-col, default Cano_label) are forced to probability 1.0 instead
of being scored by the tree - the tree was only ever trained on c/s rows,
so this matches the training protocol and keeps the shipped threshold valid.
Pass --no-override to score every row with the tree alone. Pass --threshold
to use a different operating point than the shipped 5%-FPR one.

--info prints the full model card (features, threshold, hyperparameters,
override rule, training CV metrics) without scoring anything:

```bash
python3 predict.py --info
```

## Demo - segmentation and predict.py on new data

demo/ takes github_en_miscue_v2/demo/demo_input_english.csv (the
12-utterance, 890-row demo shipped with the earlier github_en_miscue_v2
repo) and runs it through segmentation and the trained model, end to end,
using the exact same, unmodified segment_features.py and predict.py:

```bash
cd demo
conda activate w2v
python3 make_demo_input.py               # renames columns to this pipeline's convention

sbatch run_demo_segfeat_slurm.sh         # STEP 1, GPU -> demo_segfeat_v1.csv

cd ../classification
python3 predict.py --input ../demo/demo_segfeat_v1.csv \
    --output ../demo/demo_scored.csv --slim
```

make_demo_input.py only renames demo_input_english.csv's columns (label ->
Cano_label, start_time/end_time -> Cano_start_sec/Cano_end_sec, GT_binary ->
GT_binary_label) - no fold column is added, because scoring with the shipped
model needs none.

Result on the demo (890 rows, 12 utterances, 73 GT-positive):

```
threshold  : 0.858850  (5% FPR)
override   : Cano_label in ['s', 'd'] -> prob 1.0  [from Cano_label]
             23 rows forced to prob 1.0 (2.58% of input)
flagged    : 66 S (miscue) / 824 C (correct)   -> 7.42% flagged
```

This is a mechanics demo, not an evaluation - 12 utterances is far too small
to be a meaningful test set. What it confirms is that predict.py runs
unmodified on a differently-shaped, unseen input file and produces sane
miscue calls (comparing miscue_label to GT_binary_label in
demo_scored.csv: 39/73 true positives, 27/817 false positives).
