#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segment-level wav2vec2 features for reading-miscue analysis.  One language at a
time, selected with --lang.

Given a CSV where each row is a time segment (a word, usually) carrying a start
and an end time in seconds, and a wav2vec2 CTC model, emit per row:


One forward pass per utterance; every segment window is sliced out of the same
logits.

    python segment_features.py --lang hi \
        --csv segments.csv --out segments_features.csv \
        --model /path/to/model

    python segment_features.py --lang en \
        --csv segments.csv --out out.csv --model /path/to/ckpt \
        --wav-scp wav.scp --utt-col WavFileName

    python segment_features.py --build-matrices \
        --hindi-master master.csv --english-glob 'grade_*_lm_analysis.csv'

FRAME_DURATION = 0.02 s (20 ms), tau = 3, t = 0.25,
LAPLACE_SMOOTH = 0.1, EPS = 1e-8.  Greedy CTC, no LM, no sampling -- nothing is
seeded because nothing is random.

CTC decoding collapses repeated frames FIRST and drops the blank SECOND.  The
reverse order can never emit a doubled phone, which matters for Hindi geminates
(hh ii m m aa t).  Blank is always read from tokenizer.pad_token_id, never
assumed to be 0 -- for these vocabularies id 0 is a real symbol ('aa' in Hindi,
'*' in English) and the blank is the last id.

"""
import argparse
import csv
import glob as _glob
import math
import os
import sys
from collections import defaultdict

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

REPO_HOME      = os.path.dirname(os.path.abspath(__file__))
MATRICES       = os.path.join(REPO_HOME, "matrices")
FRAME_DURATION = 0.02
SAMPLE_RATE    = 16000
EPS_SYMBOL     = "<eps>"
LAPLACE_SMOOTH = 0.1
EPS            = 1e-8
NORM_CLIP_PCTL = 99.0
DROP_TOKENS    = ("*", "SIL")

LANGS = {
    "hi": {"tag": "HI", "npz": os.path.join(MATRICES, "confusion_hindi_expt.npz"),
           "env": "W2V_MODEL_HI"},
    "en": {"tag": "EN", "npz": os.path.join(MATRICES, "confusion_english_expt.npz"),
           "env": "W2V_MODEL_EN"},
}

# ============================================================================
# BEGIN VERBATIM COPY  (word_features_extraction_..._33_features.py L22-L157)
# ============================================================================
def raw_probabilities(logits):
    # Fixed: Softmax over the vocab dimension
    return torch.nn.functional.softmax(logits, dim=-1) 

def temprature_scaled_probabilities(logits, tau=3):
    # Fixed: log_softmax and softmax over the vocab dimension
    log_softmax_logits = torch.nn.functional.log_softmax(logits, dim=-1) 
    temprature_log_softmax_logits = log_softmax_logits / tau
    return torch.nn.functional.softmax(temprature_log_softmax_logits, dim=-1)

def log_max_p(probs, eps=1e-12):
    # Fixed: Clamping added to prevent log(0) -> -inf
    return torch.log(probs.clamp(min=eps)).max(dim=-1).values

def entropy(probs, eps=1e-12):
    # Fixed: Clamping added to prevent log(0) -> -inf
    safe_probs = probs.clamp(min=eps)
    return -torch.sum(probs * torch.log(safe_probs), dim=-1)

def nvidia_get_confidence_measure_bank():
    """Generate a dictionary with confidence measure functionals."""
    neg_entropy_gibbs = lambda x: (x.exp() * x).sum(-1)
    neg_entropy_alpha = lambda x, t: (x * t).exp().sum(-1)
    neg_entropy_alpha_gibbs = lambda x, t: ((x * t).exp() * x).sum(-1)
    
    def entropy_tsallis_exp(x, v, t):
        exp_neg_max_ent = math.exp((1 - math.pow(v, 1 - t)) / (1 - t))
        return (((1 - neg_entropy_alpha(x, t)) / (1 - t)).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    def entropy_gibbs_exp(x, v, t):
        exp_neg_max_ent = math.pow(v, -t * math.pow(v, 1 - t))
        return ((neg_entropy_alpha_gibbs(x, t) * t).exp() - exp_neg_max_ent) / (1 - exp_neg_max_ent)

    entropy_gibbs_lin_baseline = lambda x, v: 1 + neg_entropy_gibbs(x) / math.log(v)
    entropy_gibbs_exp_baseline = lambda x, v: (neg_entropy_gibbs(x).exp() * v - 1) / (v - 1)
    
    confidence_measure_bank = {}
    confidence_measure_bank["normalized_max_prob"] = (
        lambda x, v, t: (x.max(dim=-1)[0].exp() * v - 1) / (v - 1)
        if t == 1.0
        else ((x.max(dim=-1)[0] * t).exp() * math.pow(v, t) - 1) / (math.pow(v, t) - 1)
    )
    confidence_measure_bank["entropy_gibbs_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha_gibbs(x, t) / math.log(v) / math.pow(v, 1 - t)
    )
    confidence_measure_bank["entropy_gibbs_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_gibbs_exp(x, v, t)
    )
    confidence_measure_bank["entropy_tsallis_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + (1 - neg_entropy_alpha(x, t)) / (math.pow(v, 1 - t) - 1)
    )
    confidence_measure_bank["entropy_tsallis_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v) if t == 1.0 else entropy_tsallis_exp(x, v, t)
    )
    confidence_measure_bank["entropy_renyi_lin"] = (
        lambda x, v, t: entropy_gibbs_lin_baseline(x, v)
        if t == 1.0
        else 1 + neg_entropy_alpha(x, t).log2() / (t - 1) / math.log(v, 2)
    )
    confidence_measure_bank["entropy_renyi_exp"] = (
        lambda x, v, t: entropy_gibbs_exp_baseline(x, v)
        if t == 1.0
        else (neg_entropy_alpha(x, t).pow(1 / (t - 1)) * v - 1) / (v - 1)
    )
    return confidence_measure_bank

def min_aggregation(token_confidence_score):
    return torch.min(token_confidence_score)

def sum_aggregation(token_confidence_score):
    return torch.sum(token_confidence_score)

def mean_aggregation(token_confidence_score):
    return torch.mean(token_confidence_score)

def aggregation_combination(token_score):
    return [min_aggregation(token_score).item(), sum_aggregation(token_score).item(), mean_aggregation(token_score).item()]

def feature_extraction_aggregation_combination(logits):
    # logits shape must be [frames, vocab]
    probabilities = raw_probabilities(logits)
    temprature_probabilities = temprature_scaled_probabilities(logits, tau=3)

    log_max_f_tokens = log_max_p(probabilities)   
    log_max_f_temprature_tokens = log_max_p(temprature_probabilities)

    entropy_f_tokens = entropy(probabilities)
    entropy_f_temprature_tokens = entropy(temprature_probabilities)

    confidence_measure_bank = nvidia_get_confidence_measure_bank()

    v = logits.shape[-1]
    t = 0.25  

    # Fixed: log_softmax over the vocab dimension
    log_softmax_logits = torch.nn.functional.log_softmax(logits, dim=-1) 

    max_prob_entropy = confidence_measure_bank["normalized_max_prob"](log_softmax_logits, v, t)
    gibbs_lin_entropy = confidence_measure_bank["entropy_gibbs_lin"](log_softmax_logits, v, t)
    gibbs_exp_entropy = confidence_measure_bank["entropy_gibbs_exp"](log_softmax_logits, v, t)
    tsallis_lin_entropy = confidence_measure_bank["entropy_tsallis_lin"](log_softmax_logits, v, t)
    tsallis_exp_entropy = confidence_measure_bank["entropy_tsallis_exp"](log_softmax_logits, v, t)
    renyi_lin_entropy = confidence_measure_bank["entropy_renyi_lin"](log_softmax_logits, v, t)
    renyi_exp_entropy = confidence_measure_bank["entropy_renyi_exp"](log_softmax_logits, v, t)

    all_extraction_feature = [
        log_max_f_tokens, log_max_f_temprature_tokens,
        entropy_f_tokens, entropy_f_temprature_tokens,
        max_prob_entropy,
        gibbs_lin_entropy, gibbs_exp_entropy,
        tsallis_lin_entropy, tsallis_exp_entropy,
        renyi_lin_entropy, renyi_exp_entropy
    ]    
    
    all_combinations = []
    for feature in all_extraction_feature:
        all_combinations += aggregation_combination(feature)

    return all_combinations

# Generate the exact 33 feature names for the CSV headers
FEATURES_NAME_BASE = [
    'raw_logmax', 'temprature_logmax',
    'raw_entropy', 'temprature_entropy',
    'normalized_max_prob',
    'entropy_gibbs_lin', 'entropy_gibbs_exp',
    'entropy_tsallis_lin', 'entropy_tsallis_exp',
    'entropy_renyi_lin', 'entropy_renyi_exp'
]
ALL_FEATURES_NAME = []
for fname in FEATURES_NAME_BASE:
    ALL_FEATURES_NAME.extend([f"{fname}_min", f"{fname}_sum", f"{fname}_mean"])
# ============================================================================
# END VERBATIM COPY
# ============================================================================


# ============================================================================
# EDIT DISTANCE  (behaviour-identical to classification_edit_and_features_final
# .py [v4]).  s = hypothesis, t = reference.
#   'c' match | 's' substitution | 'i' extra symbol in hyp | 'd' ref not produced
#   insertion cost = cost[eps][hyp]   deletion cost = cost[ref][eps]
#   substitution   = cost[ref][hyp]   tie-break: diagonal > insertion > deletion
# ============================================================================
def parse_phones(phone_str):
    if pd.isna(phone_str) or str(phone_str).strip() == "":
        return []
    return str(phone_str).strip().split()


def editdistance(s, t, apply_sub_cost=False, cost_matrix=None,
                 labels=None, eps_idx=None, oov_stats=None):
    """oov_stats: optional dict counting phones absent from `labels`.  Those fall
    back to unit cost, which silently turns 'costed' into plain Levenshtein --
    v4 [fix 6] counts them rather than swallowing them."""
    n, m = len(s), len(t)
    use = (apply_sub_cost and cost_matrix is not None
           and labels is not None and eps_idx is not None)

    def resolve(seq):
        out = []
        for x in seq:
            i = labels.get(x)
            if i is None and oov_stats is not None:
                oov_stats["oov"] += 1
                oov_stats["symbols"][x] = oov_stats["symbols"].get(x, 0) + 1
            out.append(i)
        if oov_stats is not None:
            oov_stats["total"] += len(seq)
        return out

    s_idx = resolve(s) if use else [None] * n
    t_idx = resolve(t) if use else [None] * m

    if use:
        ins_cost = [cost_matrix[eps_idx][i] if i is not None else 1.0 for i in s_idx]
        del_cost = [cost_matrix[j][eps_idx] if j is not None else 1.0 for j in t_idx]
    else:
        ins_cost, del_cost = [1.0] * n, [1.0] * m

    dist = [[0.0] * (m + 1) for _ in range(n + 1)]
    op = [[None] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dist[i][0] = dist[i - 1][0] + ins_cost[i - 1]; op[i][0] = 'i'
    for j in range(1, m + 1):
        dist[0][j] = dist[0][j - 1] + del_cost[j - 1]; op[0][j] = 'd'

    for i in range(1, n + 1):
        si, sii, ic = s[i - 1], s_idx[i - 1], ins_cost[i - 1]
        for j in range(1, m + 1):
            tj, tji = t[j - 1], t_idx[j - 1]
            if si == tj:
                sub, dop = 0.0, 'c'
            else:
                sub = (cost_matrix[tji][sii]
                       if use and sii is not None and tji is not None else 1.0)
                dop = 's'
            best, bop = dist[i - 1][j - 1] + sub, dop
            if dist[i - 1][j] + ic < best:
                best, bop = dist[i - 1][j] + ic, 'i'
            if dist[i][j - 1] + del_cost[j - 1] < best:
                best, bop = dist[i][j - 1] + del_cost[j - 1], 'd'
            dist[i][j], op[i][j] = best, bop

    i, j, path = n, m, []
    while i > 0 or j > 0:
        o = op[i][j]; path.append(o)
        if o in ('c', 's'): i, j = i - 1, j - 1
        elif o == 'i':      i -= 1
        else:               j -= 1
    path.reverse()
    return dist[n][m], path


def _row_normalise(confusion, smoothing=LAPLACE_SMOOTH):
    sm = confusion + smoothing
    rs = sm.sum(axis=1, keepdims=True); rs[rs == 0] = 1
    return sm / rs


def cost_neglog(confusion, smoothing=LAPLACE_SMOOTH):
    return -np.log(_row_normalise(confusion, smoothing) + EPS)


def cost_linear(confusion, smoothing=LAPLACE_SMOOTH):
    return 1.0 - _row_normalise(confusion, smoothing)


def cost_neglog_norm(confusion, smoothing=LAPLACE_SMOOTH):
    raw, obs = cost_neglog(confusion, smoothing), confusion > 0
    if obs.sum() < 2:
        lo, hi = raw.min(), raw.max()
    else:
        v = raw[obs]; lo, hi = v.min(), np.percentile(v, NORM_CLIP_PCTL)
    return np.zeros_like(raw) if hi - lo < 1e-12 else np.clip((raw - lo) / (hi - lo), 0.0, 1.0)


def build_confusion_counts(pairs):
    """pairs: (hyp_phones, ref_phones). -> confusion[ref][hyp] with an <eps> row/col."""
    allp = set()
    for hyp, ref in pairs:
        allp.update(hyp); allp.update(ref)
    allp.discard(EPS_SYMBOL)
    phones = sorted(allp) + [EPS_SYMBOL]
    labels = {p: i for i, p in enumerate(phones)}
    eps_idx = labels[EPS_SYMBOL]
    conf = np.zeros((len(phones), len(phones)), dtype=np.float64)
    for hyp, ref in pairs:
        if not ref or not hyp:
            continue
        _, path = editdistance(hyp, ref, apply_sub_cost=False)
        h = r = 0
        for o in path:
            if o in ('c', 's'):
                if h < len(hyp) and r < len(ref):
                    hi, ri = labels.get(hyp[h]), labels.get(ref[r])
                    if hi is not None and ri is not None:
                        conf[ri][hi] += 1
                h += 1; r += 1
            elif o == 'i':
                if h < len(hyp):
                    hi = labels.get(hyp[h])
                    if hi is not None: conf[eps_idx][hi] += 1
                h += 1
            else:
                if r < len(ref):
                    ri = labels.get(ref[r])
                    if ri is not None: conf[ri][eps_idx] += 1
                r += 1
    return conf, labels, phones, eps_idx


# ============================================================================
# RUN
# ============================================================================
def ctc_decode(ids, blank, inv):
    """Collapse repeats FIRST, then drop blank."""
    coll = [k for j, k in enumerate(ids) if j == 0 or k != ids[j - 1]]
    return [inv[k] for k in coll if k != blank]


def _to_float(x):
    """'' / None / nan / 'nan' -> None; else float or None."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "none", "na", "<na>"):
        return None
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def run(a):
    spec = LANGS[a.lang]
    tag = spec["tag"]
    model_path = a.model or os.environ.get(spec["env"], "")
    if not model_path or not os.path.isdir(model_path):
        sys.exit(f"[{tag}] FATAL: no checkpoint. Pass --model or set {spec['env']}.\n"
                 f"  got: {model_path!r}")
    npz_path = a.matrix or spec["npz"]

    z = np.load(npz_path, allow_pickle=True)
    cmat = z["cost_neglog"]
    phones = list(z["phones"])
    labels = {p: i for i, p in enumerate(phones)}
    eps_idx = int(z["eps_idx"])
    print(f"[{tag}] cost matrix {cmat.shape} from {os.path.basename(npz_path)} "
          f"(fit on {int(z['n_pairs'])} pairs, laplace={float(z['laplace'])})")

    with open(a.csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows, headers = list(reader), reader.fieldnames
    print(f"[{tag}] {len(rows)} rows from {os.path.basename(a.csv)}")

    missing = [c for c in (a.utt_col, a.start_col, a.end_col) if c not in headers]
    if missing:
        sys.exit(f"[{tag}] FATAL: column(s) {missing} not in CSV.\n"
                 f"  available: {headers[:25]}{' ...' if len(headers) > 25 else ''}")
    canon_col = a.canon_col if (a.canon_col and a.canon_col.lower() != "none") else None
    if canon_col and canon_col not in headers:
        sys.exit(f"[{tag}] FATAL: --canon-col {canon_col!r} not in CSV. "
                 f"Pass --canon-col none to skip edit distance.")

    wav_map = {}
    if a.wav_scp:
        with open(a.wav_scp) as f:
            for line in f:
                p = line.strip().split(maxsplit=1)
                if len(p) == 2:
                    wav_map[p[0]] = p[1]
        print(f"[{tag}] wav.scp: {len(wav_map)} entries")

    def resolve(u):
        if u in wav_map:
            return wav_map[u]
        return u if os.path.isabs(u) and os.path.exists(u) else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path).to(device).eval()
    torch.set_grad_enabled(False)
    blank = processor.tokenizer.pad_token_id
    inv = {i: t for t, i in processor.tokenizer.get_vocab().items()}
    print(f"[{tag}] model {os.path.basename(model_path)} | vocab={len(inv)} "
          f"blank={blank} | device={device}")

    new_cols = ([f"{tag}_phone_seq_raw", f"{tag}_phone_seq_clean",
                 "frame_start", "frame_end", "n_frames"] + ALL_FEATURES_NAME)
    if canon_col:
        new_cols += ["costed_neglog", "lev_dist", "n_canon_phones", "n_hyp_phones"]
    for r in rows:
        for c in new_cols:
            r[c] = ""

    utt_to_idx = defaultdict(list)
    for i, r in enumerate(rows):
        utt_to_idx[r[a.utt_col]].append(i)

    n_ok = n_noaudio = n_nospan = n_costed = 0
    oov_stats = {"oov": 0, "total": 0, "symbols": {}}
    for utt, idxs in tqdm(utt_to_idx.items(), desc=f"{tag} utterances"):
        wp = resolve(utt)
        if wp is None:
            n_noaudio += len(idxs)
            for i in idxs:
                rows[i][f"{tag}_phone_seq_raw"] = "__NO_AUDIO__"
            continue
        try:
            speech, _ = librosa.load(wp, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"[{tag}] LOAD FAIL {utt}: {e}")
            for i in idxs:
                rows[i][f"{tag}_phone_seq_raw"] = "__LOAD_FAIL__"
            continue

        iv = processor(speech, sampling_rate=SAMPLE_RATE,
                       return_tensors="pt").input_values.to(device)
        logits = model(iv).logits
        total_frames = logits.shape[1]

        for i in idxs:
            r = rows[i]
            s, e = _to_float(r.get(a.start_col)), _to_float(r.get(a.end_col))
            if s is None or e is None:
                n_nospan += 1
                continue
            fs = max(0, math.floor(s / FRAME_DURATION))
            fe = min(total_frames, math.ceil(e / FRAME_DURATION))
            r["frame_start"], r["frame_end"] = fs, fe
            if fe <= fs:
                r["n_frames"] = 0
                n_nospan += 1
                continue
            r["n_frames"] = fe - fs

            seg = logits[:, fs:fe, :]
            toks = ctc_decode(torch.argmax(seg, dim=-1)[0].tolist(), blank, inv)
            r[f"{tag}_phone_seq_raw"] = " ".join(toks)
            hyp = [t for t in toks if t not in DROP_TOKENS]
            r[f"{tag}_phone_seq_clean"] = " ".join(hyp)

            for name, val in zip(ALL_FEATURES_NAME,
                                 feature_extraction_aggregation_combination(seg.squeeze(0).cpu())):
                r[name] = val

            if canon_col:
                ref = parse_phones(r.get(canon_col))
                r["n_canon_phones"], r["n_hyp_phones"] = len(ref), len(hyp)
                # <UNK> canonical = OOV word, no real target -> no meaningful distance
                if ref and hyp and "<UNK>" not in ref:
                    cd, _ = editdistance(hyp, ref, apply_sub_cost=True, cost_matrix=cmat,
                                         labels=labels, eps_idx=eps_idx,
                                         oov_stats=oov_stats)
                    ld, _ = editdistance(hyp, ref, apply_sub_cost=False)
                    r["costed_neglog"], r["lev_dist"] = cd, ld
                    n_costed += 1
            n_ok += 1

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers + [c for c in new_cols if c not in headers])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[{tag}] segments scored      : {n_ok}")
    print(f"[{tag}] no usable span       : {n_nospan}")
    print(f"[{tag}] no audio             : {n_noaudio}")
    print(f"[{tag}] costed_neglog filled : {n_costed}")
    if oov_stats["total"]:
        rate = 100.0 * oov_stats["oov"] / oov_stats["total"]
        top = sorted(oov_stats["symbols"].items(), key=lambda kv: -kv[1])[:8]
        print(f"[{tag}] phones unknown to the cost matrix: "
              f"{oov_stats['oov']}/{oov_stats['total']} ({rate:.2f}%)")
        if rate > 1.0:
            print(f"[{tag}]   ^ HIGH -- these fall back to unit cost, so "
                  f"costed_neglog is drifting toward plain Levenshtein.")
            print(f"[{tag}]   Most likely --lang does not match the phone "
                  f"alphabet in your canonical column.")
            print(f"[{tag}]   worst offenders: {top}")
    print(f"[{tag}] -> {a.out}")


# ============================================================================
# BUILD MATRICES
# ============================================================================
def _clean(seq):
    return [p for p in seq if p not in ("SIL", "*", "|", "")]


def build_matrices(a):
    """EXPERIMENT-SCOPED: fit on all pairs, then used to score rows from the same
    pool. costed_neglog from these is optimistic; rebuild per fold for anything
    that needs a defensible number. See README.md."""
    jobs = []
    if a.hindi_master:
        d = pd.read_csv(a.hindi_master,
                        usecols=["canonical_phone_seq", "MT_decoded_wav2vec_phone_seq_clean"])
        jobs.append(("hindi", [(_clean(parse_phones(h)), _clean(parse_phones(r)))
                               for r, h in zip(d.canonical_phone_seq,
                                               d.MT_decoded_wav2vec_phone_seq_clean)]))
    if a.english_glob:
        pairs = []
        for f in sorted(_glob.glob(a.english_glob)):
            d = pd.read_csv(f)
            for r, h in zip(d.reference_phones, d["prediction_phones_a0.0_b0.0"]):
                pairs.append((_clean(str(h).replace("|", " ").split()),
                              _clean(str(r).replace("|", " ").split())))
        jobs.append(("english", pairs))
    if not jobs:
        sys.exit("Nothing to build: pass --hindi-master and/or --english-glob.")

    os.makedirs(MATRICES, exist_ok=True)
    for name, pairs in jobs:
        pairs = [(h, r) for h, r in pairs if h and r]
        print(f"\n=== {name.upper()} ===\n  pairs: {len(pairs)}")
        conf, labels, phones, eps_idx = build_confusion_counts(pairs)
        cn = cost_neglog(conf)
        print(f"  inventory (incl <eps>): {len(phones)}")
        print(f"  counts {conf.sum():.0f}   observed cells {(conf>0).sum()}/{conf.size}")
        print(f"  cost_neglog min={cn.min():.3f} max={cn.max():.3f} "
              f"diag_mean={np.mean(np.diag(cn)):.3f}")
        out = os.path.join(MATRICES, f"confusion_{name}_expt.npz")
        np.savez(out, confusion=conf, cost_neglog=cn, cost_linear=cost_linear(conf),
                 cost_neglog_norm=cost_neglog_norm(conf),
                 phones=np.array(phones, dtype=object), eps_idx=eps_idx,
                 laplace=LAPLACE_SMOOTH, n_pairs=len(pairs))
        print(f"  -> {out}")


def main():
    p = argparse.ArgumentParser(
        description="Segment-level wav2vec2 features (one language per run).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--lang", choices=["hi", "en"], help="which language/matrix to use")
    p.add_argument("--csv", help="input CSV, one time segment per row")
    p.add_argument("--out", help="output CSV")
    p.add_argument("--model", help="wav2vec2 checkpoint dir (else $W2V_MODEL_HI/EN)")
    p.add_argument("--matrix", help="override the shipped cost matrix .npz")
    p.add_argument("--wav-scp", dest="wav_scp",
                   help="wav.scp mapping utt -> path; omit if the utt column holds a path")
    p.add_argument("--utt-col", dest="utt_col", default="utterance_id")
    p.add_argument("--start-col", dest="start_col", default="start_time")
    p.add_argument("--end-col", dest="end_col", default="end_time")
    p.add_argument("--canon-col", dest="canon_col", default="canonical_phone_seq",
                   help="canonical phones for the edit distance; 'none' to skip")
    p.add_argument("--build-matrices", action="store_true",
                   help="rebuild matrices/*.npz instead of running")
    p.add_argument("--hindi-master", help="[--build-matrices] word-level CSV")
    p.add_argument("--english-glob", help="[--build-matrices] glob of grade CSVs")
    a = p.parse_args()

    if a.build_matrices:
        return build_matrices(a)
    missing = [f"--{n}" for n in ("lang", "csv", "out") if not getattr(a, n)]
    if missing:
        p.error("missing required argument(s): " + ", ".join(missing))
    return run(a)


if __name__ == "__main__":
    sys.exit(main() or 0)
