#!/usr/bin/env python3
"""Reduce the training manifest to a self-contained export bundle description.
Run after train_classifier.py."""
import json, os, hashlib, datetime
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = json.load(open(os.path.join(REPO, 'manifest.json')))
SHIP = 'EN_new_foldavg_ed_conf2'
sel = json.load(open(os.path.join(REPO, 'train', 'selection.json')))

def sha(rel):
    h = hashlib.sha256()
    with open(os.path.join(REPO, rel), 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

required = [man[SHIP]['file'], man['_cost_matrix']['file'],
            'predict.py', 'segment_features.py', 'manifest.json', 'requirements.txt']
man['_export_bundle'] = dict(
    ships=SHIP,
    required=required,
    checksums={p: sha(p) for p in required if p != 'manifest.json'},
    note=('Copy these paths and the bundle scores on its own. The recipient still needs '
          'segment_features.py pointed at the shipped cost matrix to produce costed_neglog '
          'on the right scale.'),
    generated=datetime.datetime.now().isoformat(timespec='seconds'))
man['_feature_selection'] = sel
json.dump(man, open(os.path.join(REPO, 'manifest.json'), 'w'), indent=2)
print(f'export bundle: ships {SHIP}')
for p in required:
    print(f'  {p}')
