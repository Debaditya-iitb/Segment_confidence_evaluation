#!/usr/bin/env python3
"""STEP 6 - choose ONE edit feature + ONE confidence feature for the 2-feature
deployable classifier.

Stage A: leave-one-fold-out CV AUC of every candidate on its own.
Stage B: pair grid, top-5 confidence features x every edit feature.

Both stages use the OVERRIDE protocol the deployed scorer uses: the model is
fitted only on ASR c/s rows, i/d rows and ASR-flagged rows are pinned to p=1.0,
and metrics are computed on the full corpus.

Edit-feature candidates are restricted to what segment_features.py actually
emits (costed_neglog, lev_dist, and lev_dist_norm derived from
n_hyp_phones / n_canon_phones) - a feature the deployed extractor cannot
produce is not deployable, whatever it scores here.
"""
import os, sys, json, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score

D   = "/home/daplab/30006664/KV_English_MTP1/Data/Baseline_Test_set/new_master_file"
M   = f"{D}/master_EN_new_v3_editdist.csv"
OUT = f"{D}/feature_selection_EN_new"
os.makedirs(OUT, exist_ok=True)

CONF_BASES = ["raw_logmax","temprature_logmax","raw_entropy","temprature_entropy",
              "normalized_max_prob","entropy_gibbs_lin","entropy_gibbs_exp",
              "entropy_tsallis_lin","entropy_tsallis_exp","entropy_renyi_lin","entropy_renyi_exp"]
CONF_STATS = ["min","sum","mean"]
CONF_NAMES = [f"{b}_{s}" for b in CONF_BASES for s in CONF_STATS]
assert len(CONF_NAMES) == 33

HP = dict(learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0,
          class_weight="balanced", random_state=42, early_stopping=True, max_iter=500)

LOG = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.append(s)

df = pd.read_csv(M, low_memory=False)
P(f"[INFO] {os.path.basename(M)}: {len(df)} rows x {df.shape[1]} cols")

y      = df["GT_binary"].astype(int).to_numpy()
fa     = df["fold"].to_numpy()
folds  = sorted(df["fold"].unique())
lab    = df["Cano_label"].to_numpy()
abin   = df["Cano_binary"].astype(int).to_numpy()
cs     = np.isin(lab, ["c", "s"])
forced = (~cs) | (abin == 1)
P(f"[INFO] rows {len(df)} | c/s {int(cs.sum())} | forced {int(forced.sum())} "
  f"| positives {int(y.sum())} ({100*y.mean():.2f}%)")

# lev_dist_norm, from columns segment_features.py also emits
n_hyp   = df["canod_decoded_w2v_sequence_clean"].fillna("").astype(str).str.split().apply(len)
n_canon = df["canonical_phone_seq"].fillna("").astype(str).str.split().apply(len)
den = np.maximum(n_hyp, n_canon).replace(0, np.nan)
df["Cano_lev_dist_norm"] = df["Cano_lev_dist"] / den

EDIT_NAMES = ["costed_neglog", "lev_dist", "lev_dist_norm"]
P(f"[INFO] edit candidates (deployable by segment_features.py): {EDIT_NAMES}")
P(f"[INFO] confidence candidates: {len(CONF_NAMES)}")

def col(name):
    return pd.to_numeric(df[f"Cano_{name}"], errors="coerce").to_numpy(dtype=float)

def cv_override(X):
    """leave-one-fold-out, fit on c/s only, force the rest to 1.0, score everything."""
    o = np.full(len(df), np.nan)
    for h in folds:
        trm = (fa != h) & cs
        tem = (fa == h) & cs
        m = HistGradientBoostingClassifier(**HP)
        m.fit(X[trm], y[trm])
        idx = np.where(tem)[0]
        o[idx] = m.predict_proba(X[idx])[:, 1]
    o_ovr = o.copy(); o_ovr[forced] = 1.0
    mm_cs  = ~np.isnan(o) & cs & ~forced
    mm_ovr = ~np.isnan(o_ovr)
    def met(yy, sc):
        fpr, tpr, thr = roc_curve(yy, sc)
        r  = float(np.interp(0.05, fpr, tpr))
        t5 = float(np.interp(0.05, fpr, thr[:len(fpr)]))
        TP = r*yy.sum(); FP = 0.05*(yy == 0).sum(); p = TP/(TP+FP)
        return dict(AUC=float(roc_auc_score(yy, sc)), AP=float(average_precision_score(yy, sc)),
                    MR_at_5FPR=100*(1-r), F1=2*p*r/(p+r), Precision=p, Recall=r, threshold=t5)
    return met(y[mm_ovr], o_ovr[mm_ovr]), met(y[mm_cs], o[mm_cs])

# ------------------------------------------------------------------ STAGE A
P("\n=========== STAGE A - every candidate on its own ===========")
t0 = time.time(); rows = []
for kind, names in (("edit", EDIT_NAMES), ("conf", CONF_NAMES)):
    for n in names:
        X = col(n).reshape(-1, 1)
        mo, mc = cv_override(X)
        rows.append(dict(kind=kind, feature=n, AUC_override=mo["AUC"], AUC_model_only=mc["AUC"],
                         AP=mo["AP"], MR_at_5FPR=mo["MR_at_5FPR"], n_nan=int(np.isnan(X).sum())))
        P(f"  {kind:4s} {n:26s} AUC(ovr) {mo['AUC']:.4f}  AUC(model) {mc['AUC']:.4f}  "
          f"AP {mo['AP']:.4f}  MR@5 {mo['MR_at_5FPR']:.2f}  [{time.time()-t0:.0f}s]")
A = pd.DataFrame(rows).sort_values(["kind", "AUC_override"], ascending=[True, False])
A.to_csv(f"{OUT}/stageA_single_feature.csv", index=False)

P("\n--- STAGE A ranking, edit features ---")
P(A[A.kind == "edit"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
P("\n--- STAGE A ranking, confidence features (top 10 of 33) ---")
P(A[A.kind == "conf"].head(10).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

TOP_CONF = A[A.kind == "conf"].head(5)["feature"].tolist()
P(f"\n[INFO] stage B grid: {EDIT_NAMES} x {TOP_CONF}")

# ------------------------------------------------------------------ STAGE B
P("\n=========== STAGE B - pair grid ===========")
rows = []
for e in EDIT_NAMES:
    xe = col(e)
    for c in TOP_CONF:
        X = np.column_stack([xe, col(c)])
        mo, mc = cv_override(X)
        rows.append(dict(edit=e, conf=c, AUC_override=mo["AUC"], AUC_model_only=mc["AUC"],
                         AP=mo["AP"], MR_at_5FPR=mo["MR_at_5FPR"], F1=mo["F1"],
                         threshold=mo["threshold"]))
        P(f"  {e:16s} + {c:26s} AUC(ovr) {mo['AUC']:.4f}  AUC(model) {mc['AUC']:.4f}  "
          f"AP {mo['AP']:.4f}  MR@5 {mo['MR_at_5FPR']:.2f}  [{time.time()-t0:.0f}s]")
B = pd.DataFrame(rows).sort_values("AUC_override", ascending=False)
B.to_csv(f"{OUT}/stageB_pair_grid.csv", index=False)
P("\n--- STAGE B ranking ---")
P(B.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best = B.iloc[0]
edit_solo = float(A[(A.kind=="edit") & (A.feature==best.edit)]["AUC_override"].iloc[0])
P(f"\nWINNER: edit={best.edit}  conf={best['conf']}")
P(f"  AUC(override) {best.AUC_override:.4f}   AUC(model only) {best.AUC_model_only:.4f}")
P(f"  the confidence feature is worth {best.AUC_override - edit_solo:+.4f} AUC over "
  f"{best.edit} alone ({edit_solo:.4f})")
json.dump(dict(edit_feature=best.edit, conf_feature=best["conf"],
               AUC_override=float(best.AUC_override), AUC_model_only=float(best.AUC_model_only),
               AP=float(best.AP), MR_at_5FPR=float(best.MR_at_5FPR),
               edit_alone_AUC=edit_solo, top_conf_considered=TOP_CONF),
          open(f"{OUT}/selection.json", "w"), indent=2)
open(f"{OUT}/selection_report.txt", "w", encoding="utf-8").write("\n".join(LOG)+"\n")
P(f"\nwrote {OUT}/stageA_single_feature.csv, stageB_pair_grid.csv, selection.json, selection_report.txt")
