# =============================================================================
# COMBINED SAME-DAY + NEXT-DAY EXPERIMENT (Experiment A) — ARCHITECTURE SELECTION
#
# Reads ONLY the ten Phase-1 summaries (House-11 results). It opens no house CSV,
# loads no model and never touches Houses 12-13.
#
# Frozen primary criterion (lower is better), per final seed s in {42, 43, 44}:
#     CombinedScore_s = 0.5 x SameDay_nRMSE_s + 0.5 x NextDay_nRMSE_s
# where nRMSE_percent = sqrt(mean((pred_pu - actual_pu)^2)) x 100 on House 11,
# pooled over all daylight points of that task's own operational schedule.
# Winner = lowest MEAN CombinedScore over the three seeds. No tie rule, no
# secondary decision criterion. MAPE_1pct and all other metrics are not used.
#
# Writes frozen_selection.json. After that file exists, Phase 1 refuses to run
# and Phase 2 may be executed.
# =============================================================================

import os, json, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = Path('/content/drive/MyDrive/Multi-House')
except ImportError:  # outside Colab: point PV_DATA_DIR at the folder with the CSVs
    DATA_DIR = Path(os.environ.get('PV_DATA_DIR', 'Multi-House'))
try:
    from IPython.display import display
except ImportError:
    display = print

EXPERIMENT_DIR = DATA_DIR / 'combined_sameday_nextday_experiment'
PHASE1_DIR = EXPERIMENT_DIR / 'phase1'
SELECTION_FILE = EXPERIMENT_DIR / 'frozen_selection.json'

ARCHITECTURES = ['lstm', 'gru', 'cnn', 'lstm_cnn', 'xgboost']
TASKS = ['same_day', 'next_day']
TASK_WEIGHTS = {'same_day': 0.5, 'next_day': 0.5}
FINAL_SEEDS = [42, 43, 44]
DEPLOYMENT_SEED = 42
PRETEST_HOUSES = list(range(1, 12))

if SELECTION_FILE.exists():
    raise RuntimeError(f'{SELECTION_FILE} already exists. The selection is frozen and is not recomputed.')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------------------------------------------------------
# Load and validate all ten Phase-1 runs.
# -----------------------------------------------------------------------------
summaries, summary_sha256 = {}, {}
missing = []
for arch in ARCHITECTURES:
    for task in TASKS:
        p = PHASE1_DIR / f'{arch}_{task}' / 'phase1_summary.json'
        if not p.exists():
            missing.append(str(p))
            continue
        with open(p) as f:
            summaries[(arch, task)] = json.load(f)
        summary_sha256[f'{arch}_{task}'] = file_sha256(p)
if missing:
    raise FileNotFoundError('Phase 1 is incomplete. Missing runs:\n' + '\n'.join(missing))

problems = []
for (arch, task), s in summaries.items():
    tag = f'{arch}_{task}'
    if s.get('architecture') != arch or s.get('task') != task:
        problems.append(f'{tag}: summary identifies itself as {s.get("architecture")}_{s.get("task")}')
    if s.get('test_houses_accessed') is not False:
        problems.append(f'{tag}: test_houses_accessed is not False')
    if s.get('houses_loaded') != PRETEST_HOUSES:
        problems.append(f'{tag}: houses_loaded = {s.get("houses_loaded")}')
    if s.get('final_seeds') != FINAL_SEEDS:
        problems.append(f'{tag}: final_seeds = {s.get("final_seeds")}')
    if s.get('deployment_seed') != DEPLOYMENT_SEED:
        problems.append(f'{tag}: deployment_seed = {s.get("deployment_seed")}')
    for seed in FINAL_SEEDS:
        v = s.get('per_seed', {}).get(str(seed), {}).get('house11_nRMSE_percent')
        if v is None or not np.isfinite(v):
            problems.append(f'{tag}: missing House-11 nRMSE for seed {seed}')

# Every run must have used identical data files and identical scalers (Houses 1-10).
ref = summaries[(ARCHITECTURES[0], TASKS[0])]
for (arch, task), s in summaries.items():
    tag = f'{arch}_{task}'
    if s['house_file_sha256'] != ref['house_file_sha256']:
        problems.append(f'{tag}: house files differ from {ARCHITECTURES[0]}_{TASKS[0]}')
    for key in ('future_scaler', 'static_scaler'):
        for stat in ('mean', 'scale'):
            if not np.allclose(s[key][stat], ref[key][stat], rtol=1e-6, atol=1e-9):
                problems.append(f'{tag}: {key}.{stat} differs (scaler must be fitted on Houses 1-10 only, identically)')

# Within each task every architecture must be scored on exactly the same House-11 windows.
for task in TASKS:
    hashes = {arch: summaries[(arch, task)]['house11_validation_schedule_sha256'] for arch in ARCHITECTURES}
    if len(set(hashes.values())) != 1:
        problems.append(f'{task}: House-11 validation schedules differ across architectures: {hashes}')

if problems:
    raise RuntimeError('Phase-1 protocol check failed:\n' + '\n'.join(problems))
print('All ten Phase-1 runs found and protocol checks passed.')

# -----------------------------------------------------------------------------
# CombinedScore per architecture and seed, then mean over seeds.
# -----------------------------------------------------------------------------
rows = []
for arch in ARCHITECTURES:
    for seed in FINAL_SEEDS:
        sd = summaries[(arch, 'same_day')]['per_seed'][str(seed)]['house11_nRMSE_percent']
        nd = summaries[(arch, 'next_day')]['per_seed'][str(seed)]['house11_nRMSE_percent']
        rows.append({
            'Architecture': arch,
            'Seed': seed,
            'SameDay_nRMSE_percent': sd,
            'NextDay_nRMSE_percent': nd,
            'CombinedScore': TASK_WEIGHTS['same_day'] * sd + TASK_WEIGHTS['next_day'] * nd,
        })
per_seed = pd.DataFrame(rows)

agg = per_seed.groupby('Architecture', sort=False).agg(
    SameDay_nRMSE_mean=('SameDay_nRMSE_percent', 'mean'),
    SameDay_nRMSE_sd=('SameDay_nRMSE_percent', lambda x: x.std(ddof=1)),
    NextDay_nRMSE_mean=('NextDay_nRMSE_percent', 'mean'),
    NextDay_nRMSE_sd=('NextDay_nRMSE_percent', lambda x: x.std(ddof=1)),
    CombinedScore_mean=('CombinedScore', 'mean'),
    CombinedScore_sd=('CombinedScore', lambda x: x.std(ddof=1)),
).reset_index()
agg = agg.sort_values('CombinedScore_mean', kind='mergesort').reset_index(drop=True)
agg.insert(0, 'Rank', np.arange(1, len(agg) + 1))

best_score = agg['CombinedScore_mean'].iloc[0]
if np.sum(agg['CombinedScore_mean'] == best_score) > 1:
    raise RuntimeError(
        'Exact tie on the mean House-11 CombinedScore. The frozen protocol has no tie rule; '
        'this needs an explicit, documented human decision.'
    )
winner = str(agg['Architecture'].iloc[0])

print('\nHOUSE-11 COMBINED ARCHITECTURE SELECTION (lower is better)')
display(agg)
print(f'\nSELECTED ARCHITECTURE: {winner}')

per_seed.to_csv(EXPERIMENT_DIR / 'selection_house11_per_seed.csv', index=False)
agg.to_csv(EXPERIMENT_DIR / 'selection_house11_ranking.csv', index=False)

selection = {
    'selected_architecture': winner,
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'criterion': 'lowest mean over seeds 42,43,44 of CombinedScore = 0.5 x SameDay_nRMSE + 0.5 x NextDay_nRMSE (House 11)',
    'nRMSE_definition': 'sqrt(mean((pred_pu - actual_pu)^2)) x 100 over all daylight points of the task operational schedule, post-processed predictions (clip >= 0, night = 0)',
    'task_weights': TASK_WEIGHTS,
    'final_seeds': FINAL_SEEDS,
    'deployment_seed': DEPLOYMENT_SEED,
    'validation_house': [11],
    'test_houses_used_for_selection': False,
    'tie_rule': None,
    'ranking': agg.to_dict(orient='records'),
    'per_seed': per_seed.to_dict(orient='records'),
    'phase1_summary_sha256': summary_sha256,
    'best_hyperparameters': {f'{a}_{t}': summaries[(a, t)]['best_hyperparameters']
                             for a in ARCHITECTURES for t in TASKS},
    'house_file_sha256_houses_01_11': ref['house_file_sha256'],
}
with open(SELECTION_FILE, 'w') as f:
    json.dump(selection, f, indent=2)
print('Wrote', SELECTION_FILE)
print('Phase 1 is now frozen. Run phase2_final_unseen_test.py next.')
