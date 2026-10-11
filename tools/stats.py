"""Shared statistics for the paired per-cell analyses (study_report.py, ablation_report.py)."""
import math
import numpy as np


def sign_test(d):
    """Exact two-sided sign test on paired differences; zeros and NaNs dropped."""
    d = np.asarray([x for x in d if x == x and x != 0]); n = len(d)
    if n == 0: return float('nan')
    w = int((d < 0).sum()); k = min(w, n - w)
    return float(min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n))


def paired_row(label, diffs, lower_better=True):
    """mean, sd over cells, wins (student/arm better), cells, sign p for a list of per-cell differences."""
    v = np.asarray([x for x in diffs if x == x], float)
    wins = int((v < 0).sum()) if lower_better else int((v > 0).sum())
    return {'comparison': label, 'cells': int(len(v)), 'mean_diff': float(v.mean()) if len(v) else float('nan'),
            'sd_over_cells': float(v.std(ddof=1)) if len(v) > 1 else float('nan'), 'wins': wins, 'sign_p': sign_test(v)}
