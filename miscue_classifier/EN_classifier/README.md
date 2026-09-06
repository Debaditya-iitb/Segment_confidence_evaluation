# EN_classifier — English reading-miscue classifier

Turns the per-word output of `segment_features.py --lang en` into a miscue decision.

Two features in, a probability and a class out:

| in | |
|---|---|
| `costed_neglog` | cost-matrix-weighted edit distance, wav2vec2 phones vs canonical phones |
| `entropy_gibbs_exp_min` | lowest-confidence frame in the word's time window |

| out | |
|---|---|
| `miscue_forced` | 1 = decided by the override, not by the model |
| `miscue_prob` | probability the word is a miscue, 0–1 |
| `miscue_label` | 1 = miscue, 0 = read correctly |
| `miscue_pred` | the same decision as a letter, `S` / `C` |

Appended as the **last four** columns; nothing else is touched.

## Use

```bash
pip install -r requirements.txt
python predict.py --input features.csv --output scored.csv
```

```
--threshold 0.75                                  override the shipped 5%-FPR threshold
--asr-label-col / --asr-binary-col                where the ASR c/s/i/d label lives (override input)
--no-override                                     score every row with the model alone
--slim --list-models --info
```

`./verify.sh` runs everything end to end and asserts 14 properties.

## The override

The model is only ever **fitted** on rows the ASR labelled `c`/`s`. Rows it labelled `i`/`d`
never appear in training, so at scoring time they are not handed to a model that has never
seen their kind — they are forced positive:

```
ASR label in {i, d}  ->  prob 1.0
ASR binary == 1      ->  prob 1.0
```

This is the same rule the training CV used, so **the shipped threshold is valid for scoring with
the override on**. It reads `Cano_label` and `Cano_binary` by default; point
`--asr-label-col` / `--asr-binary-col` at your own column names. `predict.py` fails loudly if the
override is on and neither column is present, rather than silently scoring without it.

The override is most of the operating-point gain — on the same features it moves AUC 0.8598 → 0.8890
and MR@5%FPR 64.4% → 51.6%, because deletions are miscues the acoustic features genuinely cannot
score. `--no-override` gives you the model-only behaviour.

## Models

Cross-validated on 86,980 c/s rows of KV English G3/4/5 baseline (8,264 miscues, 9.33%),
leave-one-fold-out over 4 folds, metrics computed on the full corpus with the override applied.
The shipped model is refit on all 86,980 c/s rows.

One model ships, on two features:

| model | features | AUC (override) | AUC (model only) | AP | MR@5%FPR | threshold |
|---|---|---|---|---|---|---|
| **EN_foldavg_ed_conf2** | costed_neglog + entropy_gibbs_exp_min | **0.8890** | 0.8598 | 0.5022 | 51.61 | 0.8366 |

`manifest.json` describes this model alone and carries everything needed to move the bundle
elsewhere: sha256 of the model and the matrix, the feature contract, the threshold and what it
means, the override rule, the CV protocol and metrics, pinned runtime versions, and the exact
file list under `export_bundle.required`.

The confidence feature is worth **+0.023 AUC** over the edit distance alone. It was chosen by
sweeping `costed_neglog` against each of the 33 confidence features; `entropy_gibbs_exp_min` tied
for first. `_min` aggregations beat `_sum` throughout — the least-confident frame carries the
signal, summing dilutes it.

## The cost matrix is a fold average

`matrices/confusion_english_KV_G345_foldavg.npz` is the **mean of 4 leave-one-fold-out
cost_neglog matrices**, each built on ~65,200 rows that exclude its held-out fold. No matrix
fitted on the whole corpus is shipped. Across-fold spread is small: mean cell sd 0.2272 against a
mean |cell| of 6.4807 (max sd 1.6104), so the folds broadly agree and the average is stable.

The file also carries each fold's own matrix (`cost_neglog_fold_1` … `_4`), the per-fold row and
pair counts, and `cost_neglog_std`, so the averaging can be audited — `verify.sh` step 6 re-derives
the average from the four and asserts it matches.

Note on method: the **cost matrices** are averaged, not the confusion counts. `cost_neglog` is a
non-linear transform of the counts, so `mean(f(counts_k)) != f(mean(counts_k))`; averaging counts
would collapse back to a whole-corpus matrix, which is exactly what this avoids. Each row still
influences 3 of the 4 matrices — this removes the single whole-corpus fit, it is not a
fully held-out matrix.

**`costed_neglog` is only meaningful on the scale of the matrix that produced it.** The matrix that
made your input's `costed_neglog` must be this one. Point `segment_features.py` at
`matrices/confusion_english_KV_G345_foldavg.npz` (its `CONFUSION_NPZ` / `--lang en` matrix path);
its stock `confusion_english_expt.npz` is a different scale, built from 1,218 pairs, and left 4.3%
of our tokens out of vocabulary. `predict.py` prints the required matrix on every run.

## Other notes

- Target is `GT_binary`: 1 where the human transcript disagrees with the canonical word.
- Missing features are fine — `HistGradientBoostingClassifier` handles NaN natively. The demo has
  49 such rows (all of them `d` rows, which the override catches anyway).
- MR@5%FPR ~52% — English miscue detection at a strict false-alarm budget is hard on this corpus.
  AUC is the more useful number; use `--threshold` to move along the curve.

## Files

```
predict.py          the CLI — the only thing you need to run
manifest.json       the export manifest — this model only, with checksums and provenance
models/             the .joblib artifact (model + threshold + override rule + card)
matrices/           the fold-averaged cost matrix, with the 4 fold matrices inside it
demo/               a 851-row / 12-utterance sample plus its scored output
verify.sh           end-to-end check, 14 assertions
_train.py           how the model and the averaged matrix were built
_make_manifest.py   regenerates the export manifest.json after a retrain
_make_demo.py       how the demo's costed_neglog was added
_train_log.txt      the training run's console output
requirements.txt    pinned — models are pickled under scikit-learn 1.7.2
```

Retrain with `sbatch run_train_slurm.sh`, then `python _make_manifest.py` — training writes a
manifest covering every model it built, and `_make_manifest.py` reduces it to this export bundle.

## Exporting

Copy the five paths in `manifest.json` → `export_bundle.required`:

```
models/miscue_clf_EN_foldavg_ed_conf2.joblib
matrices/confusion_english_KV_G345_foldavg.npz
predict.py
manifest.json
requirements.txt
```

That is a self-contained scorer. Verify the two checksums in the manifest after copying. The
recipient also needs `segment_features.py --lang en` pointed at the shipped matrix to produce the
two input features in the first place.
