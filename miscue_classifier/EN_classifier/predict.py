#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
English miscue classifier — turns segment_features.py output into a 0/1 decision per word.

    python predict.py --input features.csv --output scored.csv

Appends three columns to the input CSV (they are the LAST three columns of the output):
    miscue_prob   probability that the word is a miscue
    miscue_label  1 = miscue, 0 = read correctly
    miscue_pred   the same decision as a letter: "S" (miscue) or "C" (correct)

The label is `miscue_prob >= threshold`, where the threshold ships inside the model
and corresponds to a 5% false-positive rate on held-out KV English G3/4/5 data. Override it
with --threshold to trade recall against false alarms.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "models", "miscue_clf_EN_new_foldavg_ed_conf2.joblib")

def main():
    ap = argparse.ArgumentParser(description="Score segment_features.py output for reading miscues.")
    ap.add_argument("--input", help="CSV produced by segment_features.py")
    ap.add_argument("--output", help="where to write the scored CSV (default: <input>_scored.csv)")
    ap.add_argument("--model", default=DEFAULT, help="model .joblib (default: EN new-master ed_conf2)")
    ap.add_argument("--threshold", type=float, help="override the model's 5%%-FPR threshold")
    ap.add_argument("--slim", action="store_true",
                    help="also write a compact <output>_slim.csv with just the key columns")
    ap.add_argument("--asr-label-col", default="Cano_label",
                    help="column holding the ASR label c/s/i/d (override input)")
    ap.add_argument("--asr-binary-col", default="Cano_binary",
                    help="column holding the ASR 0/1 flag (override input)")
    ap.add_argument("--no-override", action="store_true",
                    help="score every row with the model alone; do NOT force i/d and ASR-flagged rows to 1.0")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--info", action="store_true", help="print the model card and exit")
    a = ap.parse_args()

    if a.list_models:
        man = json.load(open(os.path.join(HERE, "manifest.json")))
        for k, v in man.items():
            if k.startswith("_"):
                continue
            f  = v.get("file") or v["model"]["file"]
            ft = v.get("features") or v["inputs"]["features"]
            ev = v.get("evaluation", {}).get("with_override", v)
            print(f"{k:24} {f:48} feats={ft} AUC={ev.get('AUC')} MR@5%FPR={ev.get('MR_at_5FPR')}")
        return

    art = joblib.load(a.model)
    if a.info or not a.input:
        card = {k: art[k] for k in ("feature_names", "threshold_5pct_fpr", "hyperparameters",
                                    "cost_matrix_source", "target", "population", "trained_on",
                                    "language", "cost_matrix_note", "override", "built")}
        card["cv_metrics"] = art["cv_metrics"]
        print(json.dumps(card, indent=2, default=str)); return

    df = pd.read_csv(a.input)
    feats = art["feature_names"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        sys.exit(f"ERROR: input is missing {missing}.\n"
                 f"       These columns come from segment_features.py; check you passed its output CSV.\n"
                 f"       Columns present: {list(df.columns)[:12]} ...")

    print(f"cost matrix: {art['cost_matrix_source']}")
    print("             the input's costed_neglog MUST come from this same matrix "
          "(segment_features.py --lang en, CONFUSION_NPZ), or the scale will not match.")

    X = df[feats].apply(pd.to_numeric, errors="coerce").values
    n_nan = int(np.isnan(X).any(axis=1).sum())
    prob = art["model"].predict_proba(X)[:, 1]

    # ---- OVERRIDE ----------------------------------------------------------
    # The model was only ever fitted on rows the ASR labelled c/s. Rows it
    # labelled i/d, and rows it already flagged, are forced positive instead of
    # being handed to a model that never saw their kind. This is the same rule
    # the training CV used, so the shipped threshold is valid for it.
    ovr = art.get("override", {})
    n_forced = 0
    if a.no_override or not ovr.get("enabled"):
        forced = np.zeros(len(df), bool)
        ovr_msg = "DISABLED (--no-override)" if a.no_override else "not defined for this model"
    else:
        lc, bc = a.asr_label_col, a.asr_binary_col
        forced = np.zeros(len(df), bool)
        have = []
        if lc in df.columns:
            forced |= df[lc].astype(str).isin(["i", "d"]).values; have.append(lc)
        if bc in df.columns:
            forced |= (pd.to_numeric(df[bc], errors="coerce").fillna(0).astype(int) == 1).values; have.append(bc)
        if not have:
            sys.exit(f"ERROR: override is on but neither {lc!r} nor {bc!r} is in the input.\n"
                     f"       Point --asr-label-col / --asr-binary-col at your ASR label columns,\n"
                     f"       or pass --no-override to score with the model alone.\n"
                     f"       Columns present: {list(df.columns)[:12]} ...")
        prob = prob.copy(); prob[forced] = 1.0
        n_forced = int(forced.sum())
        ovr_msg = f"{ovr['rule']}  [from {', '.join(have)}]"
    df["miscue_forced"] = forced.astype(int)

    thr = a.threshold if a.threshold is not None else art["threshold_5pct_fpr"]
    df["miscue_prob"] = prob.round(6)
    df["miscue_label"] = (prob >= thr).astype(int)          # 1 = miscue, 0 = correct
    df["miscue_pred"] = np.where(df["miscue_label"] == 1, "S", "C")

    df = df[[c for c in df.columns if c not in ("miscue_forced","miscue_prob","miscue_label","miscue_pred")]
            + ["miscue_forced","miscue_prob","miscue_label","miscue_pred"]]

    out = a.output or (a.input.rsplit(".", 1)[0] + "_scored.csv")
    df.to_csv(out, index=False)
    if a.slim:
        keep = [c for c in ("utterance_id", "WavFileName", "canonical_word", "decoded_word", "start_time", "end_time",
                            "GT_label", "Cano_label", *feats,
                            "miscue_forced", "miscue_prob", "miscue_label", "miscue_pred") if c in df.columns]
        slim = out.rsplit(".", 1)[0] + "_slim.csv"
        df[keep].to_csv(slim, index=False)
        print(f"slim copy  : {slim}")
    print(f"model      : {os.path.basename(a.model)}  features={feats}")
    print(f"threshold  : {thr:.6f}" + ("  (user override)" if a.threshold is not None else "  (5% FPR)"))
    print(f"override   : {ovr_msg}")
    if n_forced:
        print(f"             {n_forced:,} rows forced to prob 1.0 "
              f"({100*n_forced/len(df):.2f}% of input)")
    print(f"rows       : {len(df):,}" + (f"   ({n_nan:,} had a missing feature — handled natively by the tree)" if n_nan else ""))
    print(f"flagged    : {int(df.miscue_label.sum()):,} S (miscue) / "
          f"{int((df.miscue_label==0).sum()):,} C (correct)   -> {100*df.miscue_label.mean():.2f}% flagged")
    print(f"written to : {out}")
    print(f"appended   : miscue_forced (1=forced by override), miscue_prob (probability), "
          f"miscue_label (1=S / 0=C), miscue_pred (S/C)  — the last 4 columns")

if __name__ == "__main__":
    main()
