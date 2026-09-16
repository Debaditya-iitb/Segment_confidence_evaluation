#!/usr/bin/env python3
"""
Train and SAVE the English decision-tree miscue classifier -- KV G3/4/5
Baseline, Cano system. This is the only script that needs the 'fold' column
(for leave-one-fold-out CV and to pick the 5%-FPR threshold); the saved model
that comes out of it needs no fold column at inference -- see predict.py.

    python3 train.py --input ../segmentation/master_EN_cano_segfeat_v1.csv \\
                      --edit costed_neglog --confidence temprature_logmax_sum

--edit and --confidence choose the two features the tree is trained on.
--edit must be a column already in the input CSV, OR the literal name
'costed_neglog_cv', which asks this script to rebuild costed_neglog itself
with a leave-one-fold-out cost matrix (recommended -- segment_features.py's
own costed_neglog uses a single fixed matrix, which is optimistic for CV
feature selection/threshold-fitting; see the README).
--confidence must be one of the 33 confidence columns segment_features.py
produces, or 'auto' to pick the CV-best one automatically.

Output: models/<name>.joblib, loadable by predict.py, containing the fitted
DecisionTreeClassifier, the two feature names, the override rule, the 5%-FPR
threshold, and the CV metrics that threshold was chosen from.
"""
import argparse, json, os, time
import numpy as np, pandas as pd, joblib
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier, export_text

HERE = os.path.dirname(os.path.abspath(__file__))
REF_IMPL = "/home/daplab/30006664/wav2vec_deb/Confusion_matrix/classification_edit_and_features_final.py"

CONF_BASES = ["raw_logmax", "temprature_logmax", "raw_entropy", "temprature_entropy",
              "normalized_max_prob", "entropy_gibbs_lin", "entropy_gibbs_exp",
              "entropy_tsallis_lin", "entropy_tsallis_exp", "entropy_renyi_lin",
              "entropy_renyi_exp"]
CONF_AGGS = ["min", "sum", "mean"]
ALL_CONF_FEATURES = [f"{b}_{a}" for b in CONF_BASES for a in CONF_AGGS]

# tuned by hparam_search.py: 96-config grid x 33 confidence features x 4 folds
# on the full 88,576-row corpus, selecting on miss rate @5%FPR with override
# on. max_depth=15 ties the unconstrained optimum (47.02% vs 46.78% miss)
# without an unbounded tree.
DEFAULT_TREE_KW = dict(max_depth=15, min_samples_leaf=100, class_weight="balanced",
                        random_state=0)


def build_costed_neglog_cv(df, hyp_col, ref_col, fold_col):
    """Leave-one-fold-out costed edit distance: for fold f, the phone confusion
    matrix is fit on every OTHER fold, then used to score fold f. No row is
    ever scored by a matrix its own fold helped build. Needs >=2 folds."""
    src = open(REF_IMPL, encoding="utf-8").read()
    cut = src.index("\ndef main(")
    ns = {"__name__": "_ref"}
    exec(compile(src[:cut], REF_IMPL, "exec"), ns)
    parse_phones, editdistance = ns["parse_phones"], ns["editdistance"]
    build_confusion_counts, cost_neglog = ns["build_confusion_counts"], ns["cost_neglog"]

    cneg = np.full(len(df), np.nan)
    for f in df[fold_col].unique():
        te = df.index[df[fold_col] == f]
        tr = df.index[df[fold_col] != f]
        conf, labels, phones_list, eps_idx = build_confusion_counts(df.loc[tr], hyp_col, ref_col)
        mat = cost_neglog(conf)
        for i in te:
            h, r = parse_phones(df.at[i, hyp_col]), parse_phones(df.at[i, ref_col])
            if not h or not r:
                continue
            cneg[i] = editdistance(h, r, True, mat, labels, eps_idx)[0]
    return cneg


def metrics(y, s, target=0.05):
    ok = np.isfinite(s)
    if ok.sum() == 0 or len(np.unique(y[ok])) < 2:
        return dict(auc=np.nan, miss=np.nan, thr=np.nan, realised_fpr=np.nan, n_scored=0)
    fpr, tpr, thr = roc_curve(y[ok], s[ok])
    miss_scorable = 1.0 - float(np.interp(target, fpr, tpr))
    n_pos_ok, n_pos_miss = int((y[ok] == 1).sum()), int((y[~ok] == 1).sum())
    miss = (miss_scorable * n_pos_ok + n_pos_miss) / max(n_pos_ok + n_pos_miss, 1)
    k = np.where(fpr <= target)[0][-1]
    return dict(auc=float(roc_auc_score(y[ok], s[ok])), miss=float(miss),
                thr=float(thr[k]), realised_fpr=float(fpr[k]), n_scored=int(ok.sum()))


def cv_score(df, edit_col, conf_col, fold_col, asr_col, label_col, override_rows, tree_kw, target_fpr):
    """4(or fewer)-fold leave-one-fold-out CV, override applied, returns metrics dict."""
    y = df[label_col].to_numpy(int)
    folds = df[fold_col].to_numpy()
    asr = df[asr_col].astype(str).str.lower()
    X = np.c_[df[edit_col].to_numpy(float), df[conf_col].to_numpy(float)]
    has_xy = np.isfinite(X).all(1)
    train_scope = asr.isin(["c", "s"]).to_numpy() & has_xy

    s = np.full(len(df), np.nan)
    for f in np.unique(folds):
        tr = train_scope & (folds != f)
        te = has_xy & (folds == f)
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        clf = DecisionTreeClassifier(**tree_kw).fit(X[tr], y[tr])
        s[te] = clf.predict_proba(X[te])[:, 1]

    s_ovr = s.copy()
    forced = asr.isin(override_rows).to_numpy()
    s_ovr[forced] = 1.0 + 1e-6
    m_ovr = metrics(y, s_ovr, target_fpr)
    m_noovr = metrics(y, s, target_fpr)
    return m_ovr, m_noovr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True,
                     help="segment_features.py output CSV; must have a 'fold' column for CV")
    ap.add_argument("--edit", default="costed_neglog_cv",
                     help="edit-distance column to train on, or 'costed_neglog_cv' (default) to "
                          "rebuild it here with leave-one-fold-out matrices")
    ap.add_argument("--confidence", default="auto",
                     help="one of the 33 confidence columns, or 'auto' (default) to pick the "
                          "CV-best one")
    ap.add_argument("--label-col", default="GT_binary_label")
    ap.add_argument("--asr-label-col", default="Cano_label")
    ap.add_argument("--fold-col", default="fold")
    ap.add_argument("--override-rows", nargs="+", default=["s", "d"])
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--ref-col", default="canonical_phone_seq")
    ap.add_argument("--hyp-col", default="EN_phone_seq_clean")
    ap.add_argument("--max-depth", type=int, default=DEFAULT_TREE_KW["max_depth"])
    ap.add_argument("--min-samples-leaf", type=int, default=DEFAULT_TREE_KW["min_samples_leaf"])
    ap.add_argument("--model-name", default="miscue_tree_EN_cano",
                     help="output file is models/<model-name>.joblib")
    a = ap.parse_args()

    tree_kw = dict(max_depth=a.max_depth, min_samples_leaf=a.min_samples_leaf,
                    class_weight=DEFAULT_TREE_KW["class_weight"], random_state=0)

    df = pd.read_csv(a.input, low_memory=False)
    print(f"[INFO] {a.input}: {len(df)} rows")
    for c in (a.label_col, a.asr_label_col, a.fold_col):
        assert c in df.columns, f"missing column {c!r}. columns: {list(df.columns)[:20]}"

    if a.edit == "costed_neglog_cv":
        print(f"[INFO] rebuilding costed_neglog with leave-one-fold-out matrices "
              f"({df[a.fold_col].nunique()} folds) ...")
        t0 = time.time()
        df["costed_neglog_cv"] = build_costed_neglog_cv(df, a.hyp_col, a.ref_col, a.fold_col)
        print(f"[INFO] done in {time.time()-t0:.0f}s")
    edit_col = a.edit
    assert edit_col in df.columns, f"edit column {edit_col!r} not found"

    if a.confidence == "auto":
        print("[INFO] --confidence auto: ranking all 33 confidence features by CV miss rate ...")
        rows = []
        for c in ALL_CONF_FEATURES:
            if c not in df.columns:
                continue
            m_ovr, _ = cv_score(df, edit_col, c, a.fold_col, a.asr_label_col, a.label_col,
                                 a.override_rows, tree_kw, a.target_fpr)
            rows.append(dict(conf=c, **m_ovr))
        rank = pd.DataFrame(rows).sort_values("miss")
        print(rank.head(8)[["conf", "auc", "miss"]].to_string(index=False,
              float_format=lambda v: f"{v:.4f}"))
        conf_col = rank.iloc[0]["conf"]
        print(f"[INFO] selected confidence feature: {conf_col}")
    else:
        conf_col = a.confidence
        assert conf_col in df.columns, f"confidence column {conf_col!r} not found"

    m_ovr, m_noovr = cv_score(df, edit_col, conf_col, a.fold_col, a.asr_label_col, a.label_col,
                               a.override_rows, tree_kw, a.target_fpr)
    print(f"\n[CV RESULT] edit={edit_col} confidence={conf_col}")
    print(f"  override=True   AUC {m_ovr['auc']:.4f}  miss@{a.target_fpr*100:.0f}%FPR "
          f"{m_ovr['miss']*100:.2f}%  threshold {m_ovr['thr']:.6f}")
    print(f"  override=False  AUC {m_noovr['auc']:.4f}  miss@{a.target_fpr*100:.0f}%FPR "
          f"{m_noovr['miss']*100:.2f}%")

    # ---- refit on ALL c/s rows for the shipped model -----------------------
    y = df[a.label_col].to_numpy(int)
    asr = df[a.asr_label_col].astype(str).str.lower()
    X = np.c_[df[edit_col].to_numpy(float), df[conf_col].to_numpy(float)]
    has_xy = np.isfinite(X).all(1)
    train_scope = asr.isin(["c", "s"]).to_numpy() & has_xy
    final = DecisionTreeClassifier(**tree_kw).fit(X[train_scope], y[train_scope])
    print(f"\n[FINAL FIT] {int(train_scope.sum())} rows, depth {final.get_depth()}, "
          f"{final.get_n_leaves()} leaves")
    imp = dict(zip([edit_col, conf_col], final.feature_importances_))
    print("  feature importance: " + ", ".join(f"{k} {v:.3f}" for k, v in imp.items()))
    if final.get_depth() <= 6:
        print("  " + export_text(final, feature_names=[edit_col, conf_col], decimals=3,
                                  show_weights=True).replace("\n", "\n  "))

    art = dict(
        model=final,
        feature_names=[edit_col, conf_col],
        threshold_5pct_fpr=m_ovr["thr"],
        target_fpr=a.target_fpr,
        hyperparameters=tree_kw,
        override=dict(enabled=True, rule=f"{a.asr_label_col} in {a.override_rows} -> prob 1.0",
                      rows=list(a.override_rows)),
        asr_label_col=a.asr_label_col,
        label_col=a.label_col,
        cv_metrics=dict(with_override=m_ovr, without_override=m_noovr),
        trained_on=dict(input=os.path.abspath(a.input), n_rows=len(df),
                         n_train_rows=int(train_scope.sum()),
                         n_folds=int(df[a.fold_col].nunique())),
        language="en",
        population="KV English G3/4/5 Baseline, Cano system",
        cost_matrix_note=(
            "edit feature was rebuilt with leave-one-fold-out cost matrices at train time "
            "(see --edit costed_neglog_cv); score new data with the SAME segment_features.py "
            "--matrix your training input used, or rebuild costed_neglog the same way, or the "
            "scale will not match."
            if edit_col == "costed_neglog_cv" else
            f"edit feature is {edit_col!r} straight from segment_features.py's own cost "
            f"matrix -- score new data with segment_features.py using the SAME --matrix file "
            f"your training input used, or the scale will not match."
        ),
        built=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    os.makedirs(f"{HERE}/models", exist_ok=True)
    out_path = f"{HERE}/models/{a.model_name}.joblib"
    joblib.dump(art, out_path)
    print(f"\nwrote {out_path}")
    print(f"score new data with: python3 predict.py --input <segment_features.py output> "
          f"--model {out_path}")


if __name__ == "__main__":
    main()
