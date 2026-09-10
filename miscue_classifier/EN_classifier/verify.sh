#!/bin/bash
# End-to-end check of the EN miscue classifier (v2). Run from the repo root.
set -e
source /home/daplab/30006664/spack/linux-nehalem/miniconda3-*/etc/profile.d/conda.sh
set +u; conda activate deb_wav2vec; set -u

echo "== 1. which models are available =="
python predict.py --list-models

echo; echo "== 2. model card (features, threshold, cost matrix, override) =="
python predict.py --info

echo; echo "== 3. score the shipped demo features =="
python predict.py --input demo/demo_english_features.csv \
                  --output /tmp/en_v2_check_scored.csv --asr-label-col label --slim

echo; echo "== 4. inspect the first rows =="
python - <<'PY'
import pandas as pd
d = pd.read_csv('/tmp/en_v2_check_scored_slim.csv')
cols = [c for c in ['canonical_word','decoded_word','label','GT_label','costed_neglog',
                    'entropy_gibbs_exp_min','miscue_forced','miscue_prob','miscue_label',
                    'miscue_pred'] if c in d.columns]
print(d[cols].head(12).round(4).to_string(index=False))
PY

echo; echo "== 5. sanity checks =="
python - <<'PY'
import pandas as pd, sys, json
f = pd.read_csv('/tmp/en_v2_check_scored.csv')
s = pd.read_csv('/tmp/en_v2_check_scored_slim.csv')
i = pd.read_csv('demo/demo_english_features.csv')
ok = True
def chk(c, msg):
    global ok
    print(('  PASS  ' if c else '  FAIL  ') + msg); ok = ok and bool(c)
chk(list(f.columns)[-4:] == ['miscue_forced','miscue_prob','miscue_label','miscue_pred'],
    'prediction columns appended last')
chk(len(f) == len(i), f'row count preserved ({len(i)}, got {len(f)})')
chk(len(f.columns) == len(i.columns) + 4, f'{len(i.columns)}+4 columns (got {len(f.columns)})')
chk(set(f.miscue_label) <= {0,1}, 'miscue_label is 0/1')
chk(set(f.miscue_pred) <= {'S','C'}, 'miscue_pred is S/C')
chk(f.miscue_prob.between(0,1).all(), 'miscue_prob in [0,1]')
chk(((f.miscue_pred=='S') == (f.miscue_label==1)).all(), 'label and pred agree')
n_id = int(f.label.isin(['i','d']).sum())
chk(int(f.miscue_forced.sum()) == n_id, f'override forced every i/d row ({n_id}, got {int(f.miscue_forced.sum())})')
chk((f.loc[f.miscue_forced==1,'miscue_prob'] == 1.0).all(), 'every forced row has prob exactly 1.0')
chk((f.loc[f.miscue_forced==1,'miscue_pred'] == 'S').all(), 'every forced row is predicted S')
chk(bool(f['costed_neglog'].notna().sum() > 0.9*len(f)), 'costed_neglog present on >90% of rows')
gt = (s.GT_label.astype(str) != 'c').map({True:'S', False:'C'})
agree = 100*(s.miscue_pred == gt).mean()
chk(agree > 85.0, f'agreement with GT_label > 85% (got {agree:.1f}%)')
sys.exit(0 if ok else 1)
PY

echo; echo "== 6. the cost matrix is a fold average, not a whole-corpus fit =="
python - <<'PY'
import numpy as np, sys
z = np.load('matrices/confusion_english_KV_G345_new_foldavg.npz', allow_pickle=True)
folds = [str(x) for x in z['fold_ids']]
avg = z['cost_neglog']
per = np.stack([z[f'cost_neglog_fold_{h}'] for h in folds])
ok = np.allclose(avg, per.mean(axis=0))
print(f'  folds averaged   : {folds}')
print(f'  rows per matrix  : {list(z["fold_train_rows"])}  (each excludes its held-out fold)')
print(('  PASS  ' if ok else '  FAIL  ') + 'shipped cost_neglog == mean of the per-fold matrices')
sys.exit(0 if ok else 1)
PY

echo; echo "== 7. checksums match the manifest =="
python - <<'PY'
import json, hashlib, sys
def sha(p):
    h = hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda: f.read(1<<20), b''): h.update(b)
    return h.hexdigest()
man = json.load(open('manifest.json'))
ok = True
n = 0
for k, v in man.items():
    if not (isinstance(v, dict) and 'file' in v and 'sha256' in v):
        continue
    got = sha(v['file']); good = got == v['sha256']
    print(('  PASS  ' if good else '  FAIL  ') + f"{v['file']}  {got[:16]}...")
    ok = ok and good; n += 1
for p, want in man['_export_bundle']['checksums'].items():
    got = sha(p); good = got == want
    print(('  PASS  ' if good else '  FAIL  ') + f"export_bundle {p}  {got[:16]}...")
    ok = ok and good; n += 1
print(f'  {n} checksums verified')
sys.exit(0 if ok else 1)
PY

echo; echo "== 8. override can be switched off =="
python predict.py --input demo/demo_english_features.csv --output /tmp/en_v2_noovr.csv \
                  --asr-label-col label --no-override | grep -E "override|flagged"
python - <<'PY'
import pandas as pd, sys
n = pd.read_csv('/tmp/en_v2_noovr.csv')
ok = (n.miscue_forced.sum() == 0)
print(('  PASS  ' if ok else '  FAIL  ') + f'--no-override forces nothing (got {int(n.miscue_forced.sum())})')
sys.exit(0 if ok else 1)
PY

echo; echo "ALL CHECKS PASSED"
