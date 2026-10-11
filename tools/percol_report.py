"""E5 report: where does the student's excess error live? Per column / per pair, by type, with the pre-registered readings.

  python tools/percol_report.py --datasets adult default news --seeds 0 1 2

Reads results/<d>v_s<k>_percol.jsonl and _percol_pairs.jsonl (tools/eval_percol.py). Each file is labelled
S1/S2/S4 (study student, solver cm, by NFE), T15/T39/T99 (teacher heun 8/20/50 steps, sigma_max 80), F4 (the E1
fair teacher at 4 NFE, tag sel); other teacher files (news REF_SPECS for E7.1) are ignored here. err is averaged over
sampling seeds within a cell, then over the training seeds (cells) — per column, the column is the unit.
Deltas: d1 = S1 - T99, d4 = S4 - T99, dT15 = T15 - T99 (pp; positive = worse than the 99-NFE teacher).
Writes results/percol_columns.csv, results/percol_pairs.csv, results/percol_bytype.csv, results/percol_fig.png.
"""
import argparse, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); os.chdir(ROOT)
import numpy as np, pandas as pd


def label(r):
    if r.get('solver') == 'cm': return f"S{int(r['nfe'])}"
    if r.get('tag') == 'sel' and r.get('solver') in ('euler', 'dpmpp2m', 'heun'): return 'F4'
    if r.get('solver') == 'heun' and int(r.get('sigma_max', 80)) == 80:
        return {8: 'T15', 20: 'T39', 50: 'T99'}.get(int(r.get('steps', -1)))
    return None


def load(path):
    if not os.path.exists(path): return pd.DataFrame()
    df = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
    df['who'] = [label(r) for r in df.to_dict('records')]
    return df[df.who.notna()]


def per_unit(frames, keys):
    """mean err per (cell, unit, who) over sampling seeds, then mean over cells; wide by who."""
    df = pd.concat(frames, ignore_index=True)
    g = df.groupby(['dataset', 'seed_cell'] + keys + ['who']).err.mean().reset_index()
    g = g.groupby(['dataset'] + keys + ['who']).err.mean().unstack('who').reset_index()
    for w in ('S1', 'S2', 'S4', 'T15', 'T39', 'T99', 'F4'):
        if w not in g: g[w] = np.nan
    g['d1'] = g.S1 - g.T99; g['d4'] = g.S4 - g.T99; g['dT15'] = g.T15 - g.T99
    return g


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--datasets', nargs='+', default=['adult', 'default', 'news'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2]); a = ap.parse_args()
    C, P = [], []
    for d in a.datasets:
        for k in a.seeds:
            A = f'{d}v_s{k}'
            c = load(f'results/{A}_percol.jsonl'); p = load(f'results/{A}_percol_pairs.jsonl')
            if not len(c): print(f'-- {A}: no percol results'); continue
            c['dataset'], c['seed_cell'] = d, k; p['dataset'], p['seed_cell'] = d, k; C.append(c); P.append(p)
    if not C: sys.exit('no percol results')
    cols = per_unit(C, ['column', 'col_type', 'is_target', 'n_unique_real'])
    pairs = per_unit(P, ['col_a', 'col_b', 'pair_type', 'involves_target'])
    cols.to_csv('results/percol_columns.csv', index=False); pairs.to_csv('results/percol_pairs.csv', index=False)
    cols['card'] = np.where(cols.col_type == 'numerical', 'numerical',
                            pd.cut(cols.n_unique_real, [0, 2, 10, np.inf], labels=['cat 2', 'cat 3-10', 'cat >10']).astype(str))
    bt = pd.concat([cols.groupby(['dataset', 'col_type', 'is_target'])[['d1', 'd4', 'dT15']].mean().reset_index().assign(unit='column'),
                    cols.groupby(['dataset', 'card'])[['d1', 'd4', 'dT15']].mean().reset_index().assign(unit='cardinality'),
                    pairs.groupby(['dataset', 'pair_type'])[['d1', 'd4', 'dT15']].mean().reset_index().assign(unit='pair')])
    bt.to_csv('results/percol_bytype.csv', index=False)
    with pd.option_context('display.width', 200, 'display.max_rows', 200, 'display.float_format', lambda x: f'{x:.2f}'):
        print(bt.to_string(index=False))
    # pre-registered readings
    from scipy.stats import spearmanr
    type_dir = []
    for d, g in cols.groupby('dataset'):
        v = g.d1.clip(lower=0).sort_values(ascending=False); top = max(1, int(round(0.2 * len(v))))
        share = v.iloc[:top].sum() / v.sum() if v.sum() > 0 else float('nan')
        worst = g.sort_values('d1', ascending=False).head(5)
        print(f'\n[{d}] (a) top 20% of columns ({top}) carry {share:.0%} of positive sum d1 -> ' + ('CONCENTRATED' if share >= 0.5 else 'DIFFUSE'))
        print('     worst columns: ' + '; '.join(f"{r.column} ({r.col_type}{', target' if r.is_target else ''}) d1 {r.d1:+.2f}" for r in worst.itertuples()))
        cm, nm = g[g.col_type == 'categorical'].d1.mean(), g[g.col_type == 'numerical'].d1.mean()
        type_dir.append(np.sign(cm - nm) if abs(cm - nm) >= 0.5 else 0)
        print(f'     (b) mean d1 categorical {cm:+.2f} vs numerical {nm:+.2f}')
        ok = g[['d1', 'dT15']].dropna(); rho = spearmanr(ok.d1, ok.dT15)[0] if len(ok) > 2 else float('nan')
        print(f'     (c) Spearman(d1, dT15) over columns = {rho:.2f} -> ' + ('solver-like (>= 0.5)' if rho >= 0.5 else 'different error mode (< 0.2)' if rho < 0.2 else 'intermediate'))
    s = [x for x in type_dir if x != 0]
    same = max(sum(1 for x in s if x > 0), sum(1 for x in s if x < 0))
    print(f'\n(b) type statement allowed: {"YES" if same >= 2 else "NO"} ({same} datasets with |cat - num| >= 0.5 pp in the same direction)')
    print('(c) "the student fails where the teacher\'s coarse sampler fails" requires rho >= 0.5 on all three datasets (see above).')
    print('(d) pairs: see unit == pair rows (default shown as descriptive only; its trend metric is excluded).')
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ds = [d for d in a.datasets if d in set(cols.dataset)]
    fig, axes = plt.subplots(len(ds), 1, figsize=(10, 2.6 * len(ds)), squeeze=False)
    for ax, d in zip(axes[:, 0], ds):
        g = cols[cols.dataset == d].sort_values('d1', ascending=False).reset_index(drop=True)
        colr = np.where(g.col_type == 'categorical', '#d95f02', '#1b9e77')
        ax.bar(range(len(g)), g.d1, color=colr, width=0.8)
        ax.plot(range(len(g)), g.dT15, 'o', mfc='none', mec='0.3', ms=4, label='teacher 15 NFE - 99 NFE')
        for i, r in g.iterrows():
            if r.is_target: ax.annotate('target', (i, r.d1), textcoords='offset points', xytext=(0, 3), ha='center', fontsize=7)
        ax.axhline(0, color='0.5', lw=0.6); ax.set_ylabel('err - T99 (pp)'); ax.set_title(f'{d}: 1-NFE student excess per column (green numerical, orange categorical)', fontsize=9)
        if len(g) <= 20: ax.set_xticks(range(len(g))); ax.set_xticklabels(g.column, rotation=60, ha='right', fontsize=7)
        else: ax.set_xlabel('columns sorted by excess (see results/percol_columns.csv)')
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig('results/percol_fig.png', dpi=160)
    print('written results/percol_columns.csv, percol_pairs.csv, percol_bytype.csv, percol_fig.png')


if __name__ == '__main__':
    main()
