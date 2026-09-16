"""One-off hyperparameter search for tree_classifier_en.py.
Fixes the edit feature (Cano_costed_neglog_cv) and, for each candidate tree
config, re-selects the best of the 33 confidence features by the same 4-fold
leave-one-fold-out CV + override protocol as tree_classifier_en.py. Reports
the configs with the lowest miss rate @5% FPR."""
import numpy as np, pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, roc_curve

M = "/home/daplab/30006664/KV_English_MTP1/Data/Baseline_Test_set/Master_file_miscue_Englsih_Baseline/tree_classifier_EN/classification/reference/master_EN_cano_segfeat_v2_editdist.csv"

df = pd.read_csv(M, low_memory=False)
y = df["GT_binary_label"].to_numpy(int)
folds = df["fold"].to_numpy()
asr = df["Cano_label"].astype(str).str.lower()
edit = df["Cano_costed_neglog_cv"].to_numpy(float)
train_scope_base = asr.isin(["c", "s"]).to_numpy()
forced = asr.isin(["s", "d"]).to_numpy()

CONF_BASES = ["raw_logmax", "temprature_logmax", "raw_entropy", "temprature_entropy",
              "normalized_max_prob", "entropy_gibbs_lin", "entropy_gibbs_exp",
              "entropy_tsallis_lin", "entropy_tsallis_exp", "entropy_renyi_lin",
              "entropy_renyi_exp"]
CONF_AGGS = ["min", "sum", "mean"]
CONFS = [f"{b}_{a}" for b in CONF_BASES for a in CONF_AGGS]

# precompute X/has_xy/train_scope for every candidate confidence feature once
PREP = {}
for c in CONFS:
    conf = df[c].to_numpy(float)
    X = np.c_[edit, conf]
    has_xy = np.isfinite(X).all(1)
    PREP[c] = (X, has_xy, train_scope_base & has_xy)

FOLD_IDS = np.unique(folds)


def metrics(y, s, target=0.05):
    ok = np.isfinite(s)
    fpr, tpr, thr = roc_curve(y[ok], s[ok])
    miss_scorable = 1.0 - float(np.interp(target, fpr, tpr))
    n_pos_ok, n_pos_miss = int((y[ok] == 1).sum()), int((y[~ok] == 1).sum())
    miss = (miss_scorable * n_pos_ok + n_pos_miss) / max(n_pos_ok + n_pos_miss, 1)
    return float(roc_auc_score(y[ok], s[ok])), float(miss)


def run(conf_col, tree_kw, override=True):
    X, has_xy, train_scope = PREP[conf_col]
    s = np.full(len(df), np.nan)
    for f in FOLD_IDS:
        tr = train_scope & (folds != f)
        te = has_xy & (folds == f)
        clf = DecisionTreeClassifier(random_state=0, **tree_kw).fit(X[tr], y[tr])
        s[te] = clf.predict_proba(X[te])[:, 1]
    if override:
        s = s.copy()
        s[forced] = 1.0 + 1e-6
    return metrics(y, s)


def best_conf_for(tree_kw, override=True):
    best = None
    for c in CONFS:
        auc, miss = run(c, tree_kw, override)
        if best is None or miss < best[1]:
            best = (c, miss, auc)
    return best


def main():
    grid = []
    for max_depth in [6, 7, 8, 9, 10, 12, 15, None]:
        for min_samples_leaf in [10, 20, 30, 50, 75, 100]:
            for class_weight in ["balanced", None]:
                grid.append(dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                  min_impurity_decrease=0.0,
                                  class_weight=class_weight))
    print(f"searching {len(grid)} tree configs x {len(CONFS)} confidence features "
          f"x {len(FOLD_IDS)} folds")

    rows = []
    for i, kw in enumerate(grid):
        conf, miss, auc = best_conf_for(kw, override=True)
        rows.append(dict(**kw, best_conf=conf, miss=miss, auc=auc))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(grid)} configs done", flush=True)

    r = pd.DataFrame(rows).sort_values("miss")
    r.to_csv("hparam_search_results.csv", index=False)
    print("\nTOP 20 by miss rate @5%FPR (override=True):")
    print(r.head(20).to_string(index=False))

    top = r.iloc[0]
    print(f"\nBEST: max_depth={top['max_depth']} min_samples_leaf={top['min_samples_leaf']} "
          f"min_impurity_decrease={top['min_impurity_decrease']} class_weight={top['class_weight']} "
          f"conf={top['best_conf']}  miss={top['miss']*100:.2f}%  auc={top['auc']:.4f}")

    # also check override=False under the winning config, for the report
    kw = dict(max_depth=top["max_depth"] if pd.notna(top["max_depth"]) else None,
              min_samples_leaf=int(top["min_samples_leaf"]),
              min_impurity_decrease=top["min_impurity_decrease"],
              class_weight=top["class_weight"] if pd.notna(top["class_weight"]) else None)
    auc_f, miss_f = run(top["best_conf"], kw, override=False)
    print(f"same config, override=False: miss={miss_f*100:.2f}% auc={auc_f:.4f}")


if __name__ == "__main__":
    main()
