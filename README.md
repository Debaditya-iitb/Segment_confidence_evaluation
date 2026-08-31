# Segment-level wav2vec2 features for reading-miscue analysis

Given a CSV where each row is a **time segment** (a word, usually) and a
wav2vec2 CTC checkpoint, this produces per row:

| output | what it is |
|---|---|
| `{HI,EN}_phone_seq_raw` / `_clean` | the phone sequence the model decodes for that window (`_clean` drops `*` and `SIL`) |
| 33 confidence features | log-max-prob, entropy, and the Gibbs / Tsallis / Rényi families, each as `min` / `sum` / `mean` — including `entropy_gibbs_exp_min` |
| `costed_neglog` | edit distance to the canonical pronunciation, with substitution and indel costs `-log P(hyp\|ref)` from a phone confusion matrix |
| `lev_dist` | plain Levenshtein, for reference |

All of it comes from **one forward pass per utterance** — segment windows are
sliced out of the same logits rather than re-running the model per word.

One script, one language per run, selected with `--lang`.

---



## Run

```bash
python segment_features.py --lang hi \
    --csv  segments.csv \
    --out  segments_with_features.csv \
    --model /path/to/hindi_checkpoint
```

English, with a `wav.scp` and a different utterance column:

```bash
python segment_features.py --lang en \
    --csv segments.csv --out out.csv \
    --model /path/to/english_checkpoint \
    --wav-scp wav.scp --utt-col WavFileName
```


| flag | default | |
|---|---|---|
| `--lang` | — | `hi` or `en`; picks the shipped cost matrix |
| `--csv` / `--out` | — | input and output CSV |
| `--model` | `$W2V_MODEL_HI` / `$W2V_MODEL_EN` | checkpoint directory |
| `--wav-scp` | none | omit if the utterance column already holds a path |
| `--utt-col` | `utterance_id` | |
| `--start-col` / `--end-col` | `start_time` / `end_time` | seconds |
| `--canon-col` | `canonical_phone_seq` | `none` to skip the edit-distance columns |
| `--matrix` | `.npz` | override the cost matrix |

### Input CSV

Only three columns are required: an utterance id, and a start and end
time in **seconds**. Every other column is carried through untouched. The
timestamps can be ground-truth, MT-decoded or canonical-decoded .

Audio resolves two ways, auto-detected: by `wav.scp` key, or — with `--wav-scp`
omitted — the utterance column is taken to hold an absolute path.


## Demo data

`demo/` holds the reference input and output for both languages, from the run
described in [Known numbers](#known-numbers-from-the-reference-run).

```
demo_input_hindi.csv      675 rows,  9 cols   10 KV Hindi utterances, word segments
demo_output_hindi.csv     675 rows, 51 cols   the same, plus phones + 33 features + costed_neglog
demo_input_english.csv    741 rows, 10 cols   10 KV English utterances
demo_output_english.csv   741 rows, 52 cols   the same, plus outputs
demo_english_wav.scp       10 lines           key -> audio path, for the English run
```



## Rebuilding the cost matrices



```bash
python segment_features.py --build-matrices \
    --hindi-master  /path/to/master_file_word_level.csv \
    --english-glob  '/path/to/grade_*_lm_analysis.csv'
```

Expected columns: Hindi `canonical_phone_seq` + `MT_decoded_wav2vec_phone_seq_clean`;
English `reference_phones` + `prediction_phones_a0.0_b0.0`, `|`-separated by word.
`SIL` / `*` / `|` are stripped and whole sequences are aligned — word counts
agree on only ~30% of English utterances, so word-by-word alignment is not
reliable.


---


## Hardcoded by design

`FRAME_DURATION = 0.02` (20 ms), `tau = 3`, `t = 0.25`, `LAPLACE_SMOOTH = 0.1`,
`EPS = 1e-8`.

Blank is always read from `processor.tokenizer.pad_token_id`, never assumed to
be id 0 — for these vocabularies id 0 is a real symbol (`aa` in Hindi, `*` in
English) and the blank is the last id.



