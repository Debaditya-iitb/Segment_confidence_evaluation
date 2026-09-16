#!/usr/bin/env python3
"""
English decision-tree miscue classifier -- scores segment_features.py output for
reading miscues. No 'fold' column needed: the model is already trained (see
train.py / models/*.joblib).

    python3 predict.py --input features.csv --output scored.csv

Appends four columns to the input CSV (the LAST four columns of the output):
    miscue_forced  1 = decided by the override, not by the tree
    miscue_prob    probability the word is a miscue, 0-1
    miscue_label   1 = miscue, 0 = read correctly
    miscue_pred    the same decision as a letter, S / C

The two features scored are whatever the loaded model was trained on
(art['feature_names']) -- there is no --edit/--confidence flag here; those
belong to train.py, which chooses and bakes the features into the model.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "models", "miscue_tree_EN_cano.joblib")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", help="CSV produced by segment_features.py")
    ap.add_argument("--output", help="where to write the scored CSV (default: <input>_scored.csv)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="model .joblib (default: shipped EN Cano tree)")
    ap.add_argument("--threshold", type=float, help="override the model's 5%%-FPR threshold")
    ap.add_argument("--slim", action="store_true",
                     help="also write a compact <output>_slim.csv with just the key columns")
    ap.add_argument("--asr-label-col", help="column holding the ASR c/s/d label (default: model's own)")
    ap.add_argument("--no-override", action="store_true",
                     help="score every row with the tree alone; do NOT force S/D rows to 1.0")
    ap.add_argument("--info", action="store_true", help="print the model card and exit")
    a = ap.parse_args()

    art = joblib.load(a.model)
    if a.info or not a.input:
        card = {k: art[k] for k in ("feature_names", "threshold_5pct_fpr", "hyperparameters",
                                     "override", "asr_label_col", "label_col", "language",
                                     "population", "cost_matrix_note", "trained_on", "built")}
        card["cv_metrics"] = art["cv_metrics"]
        print(json.dumps(card, indent=2, default=str))
        return

    df = pd.read_csv(a.input, low_memory=False)
    feats = art["feature_names"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        sys.exit(f"ERROR: input is missing {missing}.\n"
                  f"       These come from segment_features.py output (plus the edit-distance\n"
                  f"       column train.py built, e.g. 'costed_neglog_cv' or 'costed_neglog').\n"
                  f"       Columns present: {list(df.columns)[:20]} ...")

    print(f"model      : {os.path.basename(a.model)}")
    print(f"features   : {feats}")
    print(art.get("cost_matrix_note", ""))

    X = df[feats].apply(pd.to_numeric, errors="coerce").values
    unscorable = np.isnan(X).any(axis=1)
    n_nan = int(unscorable.sum())
    prob = np.full(len(df), np.nan)
    if (~unscorable).any():
        prob[~unscorable] = art["model"].predict_proba(X[~unscorable])[:, 1]
    # rows with a missing feature were never seen (with any NaN) at training
    # time -- the tree's NaN-routing direction there is arbitrary, so they are
    # left unscored here rather than silently guessed. The override (S/D rows)
    # still applies to them below; anything left NaN has no ASR/model opinion.

    # ---- OVERRIDE ------------------------------------------------------
    # The tree was only ever fitted on rows the ASR labelled c/s (see train.py).
    # Rows already flagged s/d are forced positive instead of being scored by
    # a tree that never saw their kind -- the same rule the training CV used,
    # so the shipped threshold is valid with the override on.
    ovr = art.get("override", {})
    lc = a.asr_label_col or art.get("asr_label_col", "Cano_label")
    forced = np.zeros(len(df), bool)
    if a.no_override or not ovr.get("enabled"):
        ovr_msg = "DISABLED (--no-override)" if a.no_override else "not defined for this model"
    elif lc in df.columns:
        forced = df[lc].astype(str).str.lower().isin(ovr.get("rows", ["s", "d"])).values
        prob = prob.copy()
        prob[forced] = 1.0
        ovr_msg = f"{ovr['rule']}  [from {lc}]"
    else:
        sys.exit(f"ERROR: override is on but {lc!r} is not in the input.\n"
                  f"       Point --asr-label-col at your ASR label column, or pass "
                  f"--no-override to score with the tree alone.\n"
                  f"       Columns present: {list(df.columns)[:20]} ...")
    df["miscue_forced"] = forced.astype(int)

    thr = a.threshold if a.threshold is not None else art["threshold_5pct_fpr"]
    df["miscue_prob"] = np.round(prob, 6)
    label = np.where(np.isnan(prob), -1, (prob >= thr).astype(int))
    df["miscue_label"] = label
    df["miscue_pred"] = np.select([label == 1, label == 0], ["S", "C"], default="U")

    df = df[[c for c in df.columns if c not in ("miscue_forced", "miscue_prob", "miscue_label", "miscue_pred")]
            + ["miscue_forced", "miscue_prob", "miscue_label", "miscue_pred"]]

    out = a.output or (a.input.rsplit(".", 1)[0] + "_scored.csv")
    df.to_csv(out, index=False)
    if a.slim:
        keep = [c for c in ("WavFileName", "canonical_word", "GT_word", "Cano_word", lc,
                             *feats, "miscue_forced", "miscue_prob", "miscue_label", "miscue_pred")
                if c in df.columns]
        slim = out.rsplit(".", 1)[0] + "_slim.csv"
        df[keep].to_csv(slim, index=False)
        print(f"slim copy  : {slim}")

    print(f"threshold  : {thr:.6f}" + ("  (user override)" if a.threshold is not None else "  (5% FPR)"))
    print(f"override   : {ovr_msg}")
    if forced.sum():
        print(f"             {int(forced.sum()):,} rows forced to prob 1.0 "
              f"({100*forced.sum()/len(df):.2f}% of input)")
    n_unscorable = int((df["miscue_label"] == -1).sum())
    print(f"rows       : {len(df):,}" +
          (f"   ({n_unscorable:,} unscorable -- missing feature and not caught by the "
           f"override, labelled U)" if n_unscorable else ""))
    n_s, n_c = int((df.miscue_label == 1).sum()), int((df.miscue_label == 0).sum())
    print(f"flagged    : {n_s:,} S (miscue) / {n_c:,} C (correct)   -> "
          f"{100*n_s/max(n_s+n_c,1):.2f}% flagged")
    print(f"written to : {out}")
    print("appended   : miscue_forced, miscue_prob, miscue_label, miscue_pred -- the last 4 columns")


if __name__ == "__main__":
    main()
