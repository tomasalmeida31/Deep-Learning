"""Checks the setup before any job runs; exits with code 0 only if every check passes.

Checks: Python 3.11; torch 2.x; numpy < 2; pandas 2.0.3; sdmetrics 0.11.1; the other
libraries the authors' code imports; that tools/patch_verbose.py has been applied to the
authors' two training scripts; that the authors' sampler files are unchanged (git-clean);
that the six datasets fetched by the authors' download script have been processed; and
that our main tool files are present.

Run from the root of the TabSyn clone (~/tabsyn), inside the `tabsyn` conda environment:
    python tools/check_env.py
slurm/gate_one.sbatch and slurm/study_one.sbatch run this first and stop if it fails."""
import importlib, os, subprocess, sys
sys.path.insert(0, os.getcwd())
ok = True
def chk(cond, msg):
    global ok
    print(('PASS ' if cond else 'FAIL ') + msg); ok = ok and bool(cond)
print('python', sys.version.split()[0], 'cwd', os.getcwd())
chk(sys.version_info[:2] == (3, 11), 'python 3.11')
chk(os.path.basename(os.getcwd()) == 'tabsyn' and os.path.exists('tabsyn/diffusion_utils.py'), 'cwd is the tabsyn repo root')
import torch, numpy, pandas, sklearn
print('torch', torch.__version__, 'cuda build', torch.version.cuda, 'cuda available here:', torch.cuda.is_available())
chk(torch.__version__.startswith('2.'), 'torch 2.x')
chk(numpy.__version__.startswith('1.'), f'numpy<2 (have {numpy.__version__})')
chk(pandas.__version__ == '2.0.3', f'pandas==2.0.3 (have {pandas.__version__}; 3.x breaks process_dataset.py on default)')
for mod, pin in [('sdmetrics', '0.11.1'), ('xgboost', None), ('xlrd', None), ('tomli', None), ('tomli_w', None), ('icecream', None), ('category_encoders', None), ('zero', None), ('pynvml', None), ('tqdm', None), ('scipy', None)]:
    try:
        m = importlib.import_module(mod); v = getattr(m, '__version__', '?')
        chk(pin is None or v == pin, f'import {mod} ({v})' + (f' == {pin}' if pin else ''))
    except Exception as e:
        chk(False, f'import {mod}: {type(e).__name__}: {e}')
try:
    import utils_train  # pulls src -> zero, tomli, icecream
    chk(True, 'import utils_train (authors\' preprocessing chain incl. src/zero)')
except Exception as e:
    chk(False, f'import utils_train: {type(e).__name__}: {e}')
for f in ['tabsyn/vae/main.py', 'tabsyn/main.py']:
    s = open(f).read(); chk('verbose=True' not in s, f'{f}: ReduceLROnPlateau verbose kwarg removed (tools/patch_verbose.py)')
d = subprocess.run(['git', 'status', '--porcelain', '--', 'tabsyn/diffusion_utils.py', 'tabsyn/sample.py'], capture_output=True, text=True).stdout.strip()
chk(d == '', 'authors\' sampler files unmodified (git status clean for diffusion_utils.py, sample.py)' + (f' -> modified: {d}' if d else ''))
for name in ['adult', 'default', 'shoppers', 'magic', 'beijing', 'news']:
    have = all(os.path.exists(f'data/{name}/{x}') for x in ['info.json', 'X_num_train.npy', 'X_cat_train.npy', 'y_train.npy', 'X_num_test.npy', 'y_test.npy', 'train.csv', 'test.csv']) and os.path.exists(f'synthetic/{name}/real.csv')
    chk(have, f'processed dataset {name}')
for t in ['tools/make_val_split.py', 'tools/train_seeded.py', 'tools/samplers.py', 'tools/sample_grid.py', 'tools/eval_fast.py', 'tools/floor_boot.py', 'tools/gate_decide.py', 'tools/test_samplers.py', 'tools/test_decode.py', 'slurm/gate_one.sbatch']:
    chk(os.path.exists(t), f'file {t}')
print('\nCHECK_ENV', 'OK' if ok else 'FAILED'); sys.exit(0 if ok else 1)
