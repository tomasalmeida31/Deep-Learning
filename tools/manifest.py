"""Provenance manifest (review finding 20): what exactly produced the reported numbers.

  python tools/manifest.py            -> results/MANIFEST.json (+ a short printed summary)

Records: sha256 of every tools/*.py and slurm/* file (current, and the pre-review backup in _backup_2026-10-01/ when present);
sha256 + size of every teacher / VAE / student checkpoint the report uses (model.pt, encoder.pt, decoder.pt, train_z.npy, best.pt,
last.pt, DONE) for the 9 study cells and the ablation arms; split provenance (data/<alias>/split_indices.npz hash, train/val counts,
_split_fix rule); the selected configurations (tau schedules, F_k, F_1^lo, F_6); row counts and sha256 of every results/*.jsonl|csv|json;
environment (python, torch, CUDA, sdmetrics, numpy, pandas, sklearn, xgboost; `conda list` snapshot to results/conda_list.txt);
Slurm accounting for the user's jobs (sacct: job id, name, submit/start/end, elapsed, state) to results/sacct.txt; GPU-hours = sum of
elapsed for COMPLETED jobs on the GPU partition (each job holds exactly one GPU here).
"""
import glob, hashlib, json, os, platform, subprocess, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)


def sha(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(n)
            if not b: break
            h.update(b)
    return h.hexdigest()


def rec(p): return {'sha256': sha(p), 'bytes': os.path.getsize(p), 'mtime': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(os.path.getmtime(p)))} if os.path.exists(p) else None


M = {'generated': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'host': platform.node(), 'cwd': ROOT}
# code
M['code'] = {p: rec(p) for p in sorted(glob.glob('tools/*.py') + glob.glob('slurm/*') + ['tabsyn/vae/main.py', 'tabsyn/main.py', 'tabsyn/diffusion_utils.py', 'tabsyn/latent_utils.py', 'tabsyn/model.py']) if os.path.isfile(p)}
M['code_pre_review_backup'] = {p: rec(p) for p in sorted(glob.glob('_backup_2026-10-01/**/*', recursive=True)) if os.path.isfile(p)}
# checkpoints and splits
cells = [f'{d}v_s{k}' for d in ('adult', 'default', 'news') for k in (0, 1, 2)] + [f'{d}L_s0' for d in ('adult', 'default', 'news')]
M['checkpoints'], M['splits'] = {}, {}
for A in cells:
    e = {}
    for p in (f'tabsyn/vae/ckpt/{A}/encoder.pt', f'tabsyn/vae/ckpt/{A}/decoder.pt', f'tabsyn/vae/ckpt/{A}/train_z.npy', f'tabsyn/vae/ckpt/{A}/DONE', f'tabsyn/ckpt/{A}/model.pt', f'tabsyn/ckpt/{A}/DONE'):
        e[p] = rec(p)
    for sd in sorted(glob.glob(f'tabsyn/ckpt/{A}/student_*')):
        for f in ('best.pt', 'last.pt', 'ema.pt', 'student.pt', 'ref_stats.pt', 'DONE', 'log.jsonl', 'resume.jsonl'):
            p = f'{sd}/{f}'
            if os.path.exists(p): e[p] = rec(p)
        if os.path.exists(f'{sd}/DONE'):
            try: e[f'{sd}/DONE']['args'] = json.load(open(f'{sd}/DONE'))
            except Exception: pass
    M['checkpoints'][A] = e
    info = json.load(open(f'data/{A}/info.json')) if os.path.exists(f'data/{A}/info.json') else {}
    M['splits'][A] = {'train_num': info.get('train_num'), 'test_num': info.get('test_num'), 'alias_of': info.get('_alias_of'), 'split_fix': info.get('_split_fix', 'absent -> legacy_target_in_fix'),
                      'split_indices': rec(f'data/{A}/split_indices.npz'), 'real_csv': rec(f'synthetic/{A}/real.csv'), 'val_csv': rec(f'synthetic/{A}/val.csv'), 'test_csv': rec(f'synthetic/{A}/test.csv')}
# selected configurations
M['selections'] = {}
for A in cells:
    s = {}
    for f in (f'results/{A}_student_study_s{A[-1]}_taus.json', f'results/{A}_fast_choice.json', f'results/{A}_fastlo_choice.json', f'results/{A}_fast6_choice.json'):
        if os.path.exists(f): s[f] = json.load(open(f))
    M['selections'][A] = s
# results
M['results'] = {}
for p in sorted(glob.glob('results/*')):
    if os.path.isfile(p) and not p.endswith(('.png', '.pdf')):
        r = rec(p)
        if p.endswith(('.jsonl', '.csv', '.txt')):
            with open(p, 'rb') as f: r['rows'] = sum(1 for _ in f)
        M['results'][p] = r
# environment
env = {'python': sys.version, 'platform': platform.platform()}
try:
    import torch; env['torch'] = torch.__version__; env['cuda'] = torch.version.cuda; env['cudnn'] = str(torch.backends.cudnn.version())
except Exception as e: env['torch'] = f'unavailable: {e}'
for mod in ('sdmetrics', 'numpy', 'pandas', 'sklearn', 'xgboost', 'scipy'):
    try: env[mod] = __import__(mod).__version__
    except Exception as e: env[mod] = f'unavailable: {e}'
try:
    out = subprocess.run(['conda', 'list'], capture_output=True, text=True, timeout=120).stdout; open('results/conda_list.txt', 'w').write(out); env['conda_list'] = 'results/conda_list.txt'
except Exception as e: env['conda_list'] = f'unavailable: {e}'
M['environment'] = env
# slurm accounting
try:
    out = subprocess.run(['sacct', '-u', os.environ.get('USER', ''), '-S', '2026-09-01', '-X', '--format=JobID,JobName%20,Partition,Submit,Start,End,Elapsed,State,AllocTRES%40', '-P'], capture_output=True, text=True, timeout=120).stdout
    open('results/sacct.txt', 'w').write(out)
    rows = [l.split('|') for l in out.strip().splitlines()[1:]]
    def hours(e):
        d, t = (e.split('-') + [''])[:2] if '-' in e else ('0', e)
        h, m, s = (t.split(':') + ['0', '0'])[:3]; return int(d) * 24 + int(h) + int(m) / 60 + int(s) / 3600
    gpu = [r for r in rows if len(r) >= 8 and 'gpu' in r[8] and r[7].startswith('COMPLETED')]
    M['slurm'] = {'sacct_file': 'results/sacct.txt', 'jobs_total': len(rows), 'jobs_completed_with_gpu': len(gpu), 'gpu_hours_completed': round(sum(hours(r[6]) for r in gpu), 2),
                  'gpu_hours_all_states': round(sum(hours(r[6]) for r in rows if len(r) >= 8 and 'gpu' in r[8]), 2),
                  'by_name': {}}
    for r in rows:
        if len(r) >= 8 and 'gpu' in r[8]:
            n = r[1]; M['slurm']['by_name'].setdefault(n, {'jobs': 0, 'hours': 0.0}); M['slurm']['by_name'][n]['jobs'] += 1; M['slurm']['by_name'][n]['hours'] = round(M['slurm']['by_name'][n]['hours'] + hours(r[6]), 2)
except Exception as e: M['slurm'] = f'unavailable: {e}'
json.dump(M, open('results/MANIFEST.json', 'w'), indent=1)
print(f'code files {len(M["code"])}, backups {len(M["code_pre_review_backup"])}, result files {len(M["results"])}; env torch {env.get("torch")} sdmetrics {env.get("sdmetrics")}')
if isinstance(M['slurm'], dict): print(f'slurm: {M["slurm"]["jobs_total"]} jobs since 1 Sep; GPU-hours completed {M["slurm"]["gpu_hours_completed"]} (all states {M["slurm"]["gpu_hours_all_states"]}); by name: ' + ', '.join(f'{k} {v["hours"]}h/{v["jobs"]}' for k, v in sorted(M['slurm']['by_name'].items())))
for A in cells[:9]:
    s = M['splits'][A]; print(f'  {A}: train {s["train_num"]} val {s["test_num"]} split_fix={s["split_fix"]} indices={"yes" if s["split_indices"] else "MISSING"} model.pt={"yes" if M["checkpoints"][A].get(f"tabsyn/ckpt/{A}/model.pt") else "MISSING"}')
print('written results/MANIFEST.json')
