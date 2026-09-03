#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscue classifier — turns segment_features.py output into a 0/1 decision per word.

    python predict.py --input features.csv --output scored.csv

Appends three columns to the input CSV (they are the LAST three columns of the output):
    miscue_prob   probability that the word is a miscue
    miscue_label  1 = miscue, 0 = read correctly
    miscue_pred   the same decision as a letter: "S" (miscue) or "C" (correct)

The label is `miscue_prob >= threshold`, where the threshold ships inside the model
and corresponds to a 5% false-positive rate on held-out KV-1908 data. Override it
with --threshold to trade recall against false alarms.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "models", "miscue_clf_Cano_decoded_ed_conf2.joblib")

def main():
    ap = argparse.ArgumentParser(description="Score segment_features.py output for reading miscues.")
    ap.add_argument("--input", help="CSV produced by segment_features.py")
    ap.add_argument("--output", help="where to write the scored CSV (default: <input>_scored.csv)")
    ap.add_argument("--model", default=DEFAULT, help="model .joblib (default: Cano ed_conf2)")
    ap.add_argument("--threshold", type=float, help="override the model's 5%%-FPR threshold")
    ap.add_argument("--slim", action="store_true",
                    help="also write a compact <output>_slim.csv with just the key columns")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--info", action="store_true", help="print the model card and exit")
    a = ap.parse_args()

    if a.list_models:
        man = json.load(open(os.path.join(HERE, "manifest.json")))
        for k, v in man.items():
            print(f"{k:24} {v['file']:48} feats={v['features']} AUC={v['AUC']} MR@5%FPR={v['MR_at_5FPR']}")
        return

    art = joblib.load(a.model)
    if a.info or not a.input:
        card = {k: art[k] for k in ("feature_names", "threshold_5pct_fpr", "hyperparameters",
                                    "cost_matrix_source", "target", "population", "trained_on", "built")}
        card["cv_metrics"] = art["cv_metrics"]
        print(json.dumps(card, indent=2, default=str)); return

    df = pd.read_csv(a.input)
    feats = art["feature_names"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        sys.exit(f"ERROR: input is missing {missing}.\n"
                 f"       These columns come from segment_features.py; check you passed its output CSV.\n"
                 f"       Columns present: {list(df.columns)[:12]} ...")

    X = df[feats].apply(pd.to_numeric, errors="coerce").values
    n_nan = int(np.isnan(X).any(axis=1).sum())
    prob = art["model"].predict_proba(X)[:, 1]
    thr = a.threshold if a.threshold is not None else art["threshold_5pct_fpr"]
    df["miscue_prob"] = prob.round(6)
    df["miscue_label"] = (prob >= thr).astype(int)          # 1 = miscue, 0 = correct
    df["miscue_pred"] = np.where(df["miscue_label"] == 1, "S", "C")

    out = a.output or (a.input.rsplit(".", 1)[0] + "_scored.csv")
    df.to_csv(out, index=False)
    if a.slim:
        keep = [c for c in ("utterance_id", "canonical_word", "start_time", "end_time",
                            "GT_label", "costed_neglog", "entropy_gibbs_exp_min",
                            "miscue_prob", "miscue_label", "miscue_pred") if c in df.columns]
        slim = out.rsplit(".", 1)[0] + "_slim.csv"
        df[keep].to_csv(slim, index=False)
        print(f"slim copy  : {slim}")
    print(f"model      : {os.path.basename(a.model)}  features={feats}")
    print(f"threshold  : {thr:.6f}" + ("  (user override)" if a.threshold is not None else "  (5% FPR)"))
    print(f"rows       : {len(df):,}" + (f"   ({n_nan:,} had a missing feature — handled natively by the tree)" if n_nan else ""))
    print(f"flagged    : {int(df.miscue_label.sum()):,} S (miscue) / "
          f"{int((df.miscue_label==0).sum()):,} C (correct)   -> {100*df.miscue_label.mean():.2f}% flagged")
    print(f"written to : {out}")
    print(f"appended   : miscue_prob (probability), miscue_label (1=S / 0=C), miscue_pred (S/C)"
          f"  — the last 3 columns")

if __name__ == "__main__":
    main()
