# EN miscue classifier v2 — two features in, a miscue decision out

Scores the per-word output of `segment_features.py --lang en` for reading miscues in
children's English oral reading (KV Grade 3/4/5).

Two features go in:

| in | |
|---|---|
| `costed_neglog` | cost-matrix-weighted edit distance, wav2vec2 phones vs canonical phones |
| `entropy_gibbs_exp_min` | the least-confident frame in the word's time window |

Four columns come out, appended as the **last four** columns; nothing else in your CSV is touched:

| out | |
|---|---|
| `miscue_forced` | 1 = decided by the override, not by the model |
| `miscue_prob` | probability the word is a miscue, 0–1 |
| `miscue_label` | 1 = miscue, 0 = read correctly |
| `miscue_pred` | the same decision as a letter, `S` / `C` |

**Held-out AUC 0.9149**, AP 0.5805, miss rate 37.19% at a 5% false-alarm budget.

## Use — the two steps, end to end

Step 1 makes the features and needs a GPU (one wav2vec2 forward pass per utterance).
Step 2 turns them into decisions and is CPU-only, seconds.

```bash
cd /home/daplab/30006664/github_en_miscue_v2

# STEP 1 — features.  --matrix MUST be this repo's fold-averaged matrix.
python segment_features.py --lang en \
    --csv     demo/demo_input_english.csv \
    --out     demo/demo_english_features.csv \
    --model   /home/daplab/30006664/Wav2vec_models_Raj/xlsr_IITM_FT_WPP_5_FT_NSO_MID_END_CTC_LOWEST_TOP_40_hr \
    --matrix  matrices/confusion_english_KV_G345_new_foldavg.npz \
    --wav-scp demo/demo_english_wav.scp \
    --utt-col WavFileName

# STEP 2 — classify.  --asr-label-col names YOUR c/s/i/d column.
python predict.py \
    --input  demo/demo_english_features.csv \
    --output demo/demo_english_scored.csv \
    --asr-label-col label \

```

`train/run_demo_slurm.sh` is exactly the above as an sbatch job.




## The override

The model is only ever **fitted** on rows the ASR labelled `c`/`s`. Rows labelled `i`/`d` never
appear in training, so at scoring time they are not handed to a model that has never seen their
kind — they are forced positive:

```
ASR label in {i, d}  ->  prob 1.0
ASR binary == 1      ->  prob 1.0
```



## Other notes

- Target is `GT_binary`: 1 where the human transcript disagrees with the canonical word.
- Missing features are fine — `HistGradientBoostingClassifier` handles NaN natively. Rows with
  no usable time span get NaN, and they are almost all `d` rows the override catches anyway.
- Use `--threshold` to move along the ROC curve; the shipped 0.8453 is the 5%-FPR point.


