# Segment-level wav2vec2 features

Give it a CSV where each row is a **time segment** (a word, usually) and a
wav2vec2 CTC checkpoint. It appends, per row, the phones the model decodes for
that window, 33 confidence features, and the edit distance to the canonical
pronunciation.

| appended column | what it is |
|---|---|
| `{HI,EN}_phone_seq_raw` / `_clean` | the phone sequence decoded for that window (`_clean` drops `*` and `SIL`) |
| `frame_start` / `frame_end` / `n_frames` | the logit frames the window mapped to |
| 33 confidence features | log-max-prob, entropy, and the Gibbs / Tsallis / Rényi families, each as `min` / `sum` / `mean` |
| `costed_neglog` | edit distance to the canonical, substitution and indel costs `-log P(hyp\|ref)` from a phone confusion matrix |
| `lev_dist` | plain Levenshtein, for reference |
| `n_canon_phones` / `n_hyp_phones` | lengths the distance was computed over |



## Run


**Hindi** — its utterance column already holds absolute paths, so there is no
`--wav-scp`, and every column name is the default :
```bash
python segment_features.py \
    --lang hi \
    --csv   demo/demo_input_hindi.csv \
    --out   demo_output_hindi_rerun.csv \
    --model /path/to/hindi_checkpoint
```

**English** — its CSV names the utterance column `WavFileName`, so this is the
one run that needs `--utt-col`:

```bash
python segment_features.py \
    --lang en \
    --csv     demo/demo_input_english.csv \
    --out     demo_output_english_rerun.csv \
    --model   model/xlsr_IITM_FT_WPP_5 \
    --wav-scp demo/demo_english_wav.scp \
    --utt-col WavFileName
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

The reference input and output for both languages,are given as below,

```
demo_input_hindi.csv      675 rows,  9 cols   10 KV Hindi utterances, word segments
demo_output_hindi.csv     675 rows, 51 cols   the same, plus phones + 33 features + costed_neglog
demo_input_english.csv    741 rows, 10 cols   10 KV English utterances
demo_output_english.csv   741 rows, 52 cols   the same, plus outputs
demo_english_wav.scp       10 lines           key -> audio path, for the English run
```

## How the code works

One file, `segment_features.py`, one language per run, selected with `--lang`.
Per utterance:

1. Load the audio at 16 kHz and run **one forward pass** to get the logits.
2. For each of that utterance's rows, slice the window
   `[floor(start / 0.02) : ceil(end / 0.02)]` out of those logits, clamped to
   the utterance. Every segment
   is sliced from the same logits rather than re-running the model per word .
3. Greedy-decode the window: **collapse repeated frames first, drop the blank
   second.**
4. Compute the 33 confidence features over the window's frames.


Blank is always read from `processor.tokenizer.pad_token_id`, never assumed to
be id 0 — for these vocabularies id 0 is a real symbol (`aa` in Hindi, `*` in
English) and the blank is the last id.

Hardcoded by design: `FRAME_DURATION = 0.02` (20 ms), `tau = 3`, `t = 0.25`,
`LAPLACE_SMOOTH = 0.1`, `EPS = 1e-8`.

`matrices/*.npz` are the phone-confusion matrices, one per language.
They hold only aggregate phone-to-phone counts (42×42 Hindi, 40×40 English), so
no utterance, speaker or school survives the aggregation.
