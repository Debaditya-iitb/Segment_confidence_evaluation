#!/usr/bin/env python3
"""Train the deployable English miscue classifier (v2 — new merged GT/Cano timed master).

Two features go in, one probability comes out:
    <edit feature>  cost-matrix-weighted edit distance, wav2vec2 phones vs canonical phones
    <conf feature>  the least-confident frame in the word's time window
Both are chosen by train/select_features.py, which writes selection.json.

COST MATRIX = the MEAN of the per-fold leave-one-fold-out cost_neglog matrices.
Four matrices are built, each on 3 of the 4 folds, and the four are averaged.
No matrix fitted on the whole corpus is ever shipped.  The cost matrices are
averaged, not the confusion counts: cost_neglog is a non-linear transform, so
mean(f(counts_k)) != f(mean(counts_k)), and averaging counts would collapse back
to a whole-corpus fit.

OVERRIDE.  The model only ever sees rows the ASR labelled c/s.  Everything else
is forced positive, exactly as the research pipeline does:
    ASR label in {i, d}  -> prob 1.0
    ASR binary == 1      -> prob 1.0
Metrics and the 5%-FPR threshold are computed on the FULL corpus with the
override applied, so the shipped threshold is valid for how predict.py scores.
"""
import pandas as pd, numpy as np, json, os, joblib, datetime, hashlib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W    = '/home/daplab/30006664/wav2vec_deb/'
MAST = ('/home/daplab/30006664/KV_English_MTP1/Data/Baseline_Test_set/'
        'new_master_file/master_EN_new_v3_editdist.csv')
SEL  = json.load(open(os.path.join(REPO, 'train', 'selection.json')))
EDIT_FEAT, CONF_FEAT = SEL['edit_feature'], SEL['conf_feature']
NPZ_REL = 'matrices/confusion_english_KV_G345_new_foldavg.npz'
print(f'selection.json -> edit={EDIT_FEAT}  conf={CONF_FEAT}')

L = open(W + 'Confusion_matrix/classification_edit_and_features_final.py', encoding='utf-8').read().split('\n')
cut = [i for i, l in enumerate(L) if l.startswith('def main(')][0]
g = {}; exec(compile('\n'.join(L[:cut]), 'x', 'exec'), g)
P, ED, BCC, CNL = g['parse_phones'], g['editdistance'], g['build_confusion_counts'], g['cost_neglog']

usecols = ['fold', 'GT_binary', 'Cano_label', 'Cano_binary', 'canonical_phone_seq',
           'canod_decoded_w2v_sequence_clean', f'Cano_{CONF_FEAT}']
df = pd.read_csv(MAST, usecols=usecols, encoding='utf-8', low_memory=False)
df = df[df['GT_binary'].isin([0, 1])].reset_index(drop=True)
y     = df['GT_binary'].astype(int).values
fa    = df['fold'].values
folds = sorted(df['fold'].unique())
lab   = df['Cano_label'].values
abin  = df['Cano_binary'].astype(int).values
cs    = np.isin(lab, ['c', 's'])
forced = ~cs | (abin == 1)
print(f'master {len(df)} rows | c/s {int(cs.sum())} | forced {int(forced.sum())} | '
      f'positives {int(y.sum())} ({100*y.mean():.2f}%)', flush=True)

ref  = [P(v) for v in df['canonical_phone_seq'].values]
hyp  = [P(v) for v in df['canod_decoded_w2v_sequence_clean'].values]
conf = pd.to_numeric(df[f'Cano_{CONF_FEAT}'], errors='coerce').values

# ---------- per-fold matrices, then the average ----------
hyp_ref = pd.DataFrame({'h': df['canod_decoded_w2v_sequence_clean'], 'r': df['canonical_phone_seq']})
fold_cms, fold_meta, phones_ref = {}, {}, None
for h in folds:
    trm = (fa != h) & cs
    confusion, labels, phones_list, eps_idx = BCC(hyp_ref[trm], 'h', 'r')
    if phones_ref is None:
        phones_ref, eps_ref = phones_list, eps_idx
    assert phones_list == phones_ref, f'fold {h} phone list differs; cannot average'
    fold_cms[h] = CNL(confusion)
    fold_meta[h] = dict(train_rows=int(trm.sum()),
                        n_pairs=int(sum(1 for i in np.where(trm)[0] if hyp[i] and ref[i])))
    print(f'  fold {h}: matrix {fold_cms[h].shape} on {fold_meta[h]["train_rows"]} rows '
          f'({fold_meta[h]["n_pairs"]} pairs)', flush=True)

CM_AVG = np.mean(np.stack([fold_cms[h] for h in folds]), axis=0)
spread = np.stack([fold_cms[h] for h in folds]).std(axis=0)
LAB = {p: i for i, p in enumerate(phones_ref)}
npz = os.path.join(REPO, NPZ_REL)
np.savez(npz, cost_neglog=CM_AVG, cost_neglog_std=spread,
         phones=np.array(phones_ref, dtype=object), eps_idx=eps_ref, laplace=0.1,
         n_folds=len(folds), fold_ids=np.array(folds, dtype=object),
         n_pairs=int(round(np.mean([fold_meta[h]['n_pairs'] for h in folds]))),
         fold_train_rows=np.array([fold_meta[h]['train_rows'] for h in folds]),
         fold_n_pairs=np.array([fold_meta[h]['n_pairs'] for h in folds]),
         **{f'cost_neglog_fold_{h}': fold_cms[h] for h in folds})
print(f'averaged {len(folds)} fold matrices -> {npz}')
print(f'  mean |cell| {np.abs(CM_AVG).mean():.4f}  mean across-fold sd {spread.mean():.4f} '
      f'(max {spread.max():.4f})', flush=True)

# ---------- edit feature on the AVERAGED matrix (the deployment scale) ----------
st = {'oov_tokens': 0, 'total_tokens': 0}
cn = np.full(len(df), np.nan)
for i in range(len(df)):
    if hyp[i] and ref[i]:
        if EDIT_FEAT == 'costed_neglog':
            cn[i] = ED(hyp[i], ref[i], apply_sub_cost=True, cost_matrix=CM_AVG,
                       labels=LAB, eps_idx=eps_ref, oov_stats=st)[0]
        elif EDIT_FEAT == 'lev_dist':
            cn[i] = ED(hyp[i], ref[i], apply_sub_cost=False)[0]
        elif EDIT_FEAT == 'lev_dist_norm':
            cn[i] = ED(hyp[i], ref[i], apply_sub_cost=False)[0] / max(len(hyp[i]), len(ref[i]))
        else:
            raise SystemExit(f'unknown edit feature {EDIT_FEAT}')
print(f'{EDIT_FEAT} on {int(np.isfinite(cn).sum())} rows | '
      f'OOV {st["oov_tokens"]}/{st["total_tokens"]}', flush=True)

HP = dict(learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
          class_weight='balanced', random_state=42, early_stopping=True, max_iter=500)
def mk(): return HistGradientBoostingClassifier(**HP)

def met(yy, sc):
    fpr, tpr, thr = roc_curve(yy, sc); r = float(np.interp(0.05, fpr, tpr))
    t5 = float(np.interp(0.05, fpr, thr[:len(fpr)]))
    TP = r * yy.sum(); FP = 0.05 * (yy == 0).sum(); p = TP / (TP + FP)
    return dict(AUC=float(roc_auc_score(yy, sc)), AP=float(average_precision_score(yy, sc)),
                MR_at_5FPR=100 * (1 - r), F1=2 * p * r / (p + r),
                Precision=p, Recall=r, threshold=t5)

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

man = {}
for tag, names, X in (('ed1', [EDIT_FEAT], cn.reshape(-1, 1)),
                      ('ed_conf2', [EDIT_FEAT, CONF_FEAT], np.column_stack([cn, conf]))):
    o = np.full(len(df), np.nan)
    for h in folds:
        trm = (fa != h) & cs
        tem = (fa == h) & cs
        m = mk(); m.fit(X[trm], y[trm])
        idx = np.where(tem)[0]
        o[idx] = m.predict_proba(X[idx])[:, 1]
    o_ovr = o.copy(); o_ovr[forced] = 1.0
    mm_cs  = ~np.isnan(o) & cs & ~forced
    mm_ovr = ~np.isnan(o_ovr)
    M_ovr = met(y[mm_ovr], o_ovr[mm_ovr])
    M_cs  = met(y[mm_cs],  o[mm_cs])
    perfold = []
    for h in folds:
        sel = mm_ovr & (fa == h)
        perfold.append(dict(fold=int(h), n=int(sel.sum()), pos=int(y[sel].sum()),
                            AUC=round(float(roc_auc_score(y[sel], o_ovr[sel])), 4)))

    final = mk(); final.fit(X[cs], y[cs])
    key = f'EN_new_foldavg_{tag}'
    art = dict(model=final, feature_names=names, threshold_5pct_fpr=M_ovr['threshold'],
               hyperparameters=HP,
               cost_matrix_source=f'{NPZ_REL} (key cost_neglog)',
               cost_matrix_note=f'mean of {len(folds)} leave-one-fold-out cost_neglog matrices',
               override=dict(enabled=True,
                             rule="ASR label in {i,d} -> prob 1.0; ASR binary == 1 -> prob 1.0",
                             asr_label_col='Cano_label', asr_binary_col='Cano_binary',
                             n_forced_train=int(forced.sum())),
               target='1 = miscue (S), 0 = correct (C)',
               population='model fitted on ASR label in {c,s}; other rows forced positive',
               trained_on=(f'KV English G3/4/5 baseline, merged GT/Cano timed master '
                           f'(input_EN_KV_G345_GT_Cano_Merged_Timed.csv), Cano timestamps, '
                           f'{int(cs.sum())} c/s rows'),
               master_file=os.path.basename(MAST),
               language='English', cv_metrics=M_ovr, cv_metrics_model_only=M_cs,
               cv_per_fold=perfold,
               feature_selection=SEL,
               sklearn_note='pickled under scikit-learn 1.7.2',
               built=datetime.datetime.now().isoformat(timespec='seconds'))
    fp = os.path.join(REPO, 'models', f'miscue_clf_{key}.joblib')
    joblib.dump(art, fp, compress=3)
    man[key] = dict(file=f'models/miscue_clf_{key}.joblib', features=names,
                    cost_matrix=NPZ_REL, cost_matrix_kind=f'mean of {len(folds)} LOO fold matrices',
                    override=True, sha256=sha(fp),
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in M_ovr.items()},
                    AUC_model_only=round(M_cs['AUC'], 4), per_fold_AUC=perfold,
                    n_train=int(cs.sum()), n_forced=int(forced.sum()))
    print(f'  {key:24} OVERRIDE AUC {M_ovr["AUC"]:.4f}  AP {M_ovr["AP"]:.4f}  '
          f'MR@5%FPR {M_ovr["MR_at_5FPR"]:.2f}  thr {M_ovr["threshold"]:.4f}   '
          f'| model-only AUC {M_cs["AUC"]:.4f}', flush=True)

man['_cost_matrix'] = dict(file=NPZ_REL, sha256=sha(npz),
                           mean_abs_cell=round(float(np.abs(CM_AVG).mean()), 4),
                           mean_across_fold_sd=round(float(spread.mean()), 4),
                           max_across_fold_sd=round(float(spread.max()), 4),
                           n_phones=len(phones_ref))
json.dump(man, open(os.path.join(REPO, 'manifest.json'), 'w'), indent=2)
print('\n' + pd.DataFrame({k: v for k, v in man.items() if not k.startswith('_')}).T
      .drop(columns=['sha256', 'per_fold_AUC']).to_string())
best = max((k for k in man if not k.startswith('_')), key=lambda k: man[k]['AUC'])
print(f'\nBEST: {best}  AUC {man[best]["AUC"]}')
