#!/usr/bin/env python3
"""Build the demo input for the repo: 12 utterances lifted straight out of the
training master, so canonical_phone_seq comes from the same lexicon the model
was trained against.  Emits demo/demo_input_english.csv + demo/demo_english_wav.scp."""
import os, pandas as pd
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = ('/home/daplab/30006664/KV_English_MTP1/Data/Baseline_Test_set/'
     'new_master_file/master_EN_new_v1_phones.csv')
N = 12
df = pd.read_csv(M, dtype=str, keep_default_na=False)
u = df["WavFileName_original"].drop_duplicates()
sel = list(u.iloc[::max(1, len(u)//N)][:N])
d = df[df["WavFileName_original"].isin(sel)].copy()
scp = d.drop_duplicates("WavFileName_original")[["WavFileName_original", "WavFileName"]]
out = pd.DataFrame({
    "WavFileName":         d["WavFileName_original"],
    "grade":               d["grade"],
    "canonical_word":      d["canonical_word"],
    "decoded_word":        d["Cano_word"],
    "label":               d["Cano_label"],          # ASR c/s/i/d -> predict.py --asr-label-col
    "start_time":          d["Cano_start_sec"],
    "end_time":            d["Cano_end_sec"],
    "canonical_phone_seq": d["canonical_phone_seq"],
    "GT_label":            d["GT_label"],            # human truth, for the demo's own scoring
    "GT_binary":           d["GT_binary"],
})
os.makedirs(f"{REPO}/demo", exist_ok=True)
out.to_csv(f"{REPO}/demo/demo_input_english.csv", index=False, encoding="utf-8")
with open(f"{REPO}/demo/demo_english_wav.scp", "w") as f:
    for _, r in scp.iterrows():
        f.write(f"{r.WavFileName_original}\t{r.WavFileName}\n")
print(f"demo input: {len(out)} rows, {out.WavFileName.nunique()} utterances")
print(f"  label counts   : {out.label.value_counts().to_dict()}")
print(f"  GT_binary pos  : {int(out.GT_binary.astype(int).sum())} ({100*out.GT_binary.astype(int).mean():.2f}%)")
print(f"  wav.scp        : {len(scp)} entries")
