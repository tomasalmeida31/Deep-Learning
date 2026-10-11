"""Remove the `verbose=True` kwarg from ReduceLROnPlateau in the authors' two training
scripts (the kwarg was removed from torch; on torch >= 2.7 the scripts crash at start).
Idempotent and anchored: asserts the exact original text is present or already patched."""
import re, sys
targets = {
 'tabsyn/vae/main.py': ("scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)",
                        "scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10)"),
 'tabsyn/main.py':     ("scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)",
                        "scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)"),
}
for path, (old, new) in targets.items():
    src = open(path).read()
    if new in src and old not in src:
        print(f'already patched: {path}'); continue
    assert src.count(old) == 1, f'anchor not found exactly once in {path}'
    open(path, 'w').write(src.replace(old, new))
    print(f'patched: {path}')
for path, (old, new) in targets.items():
    s = open(path).read(); assert new in s and 'verbose=True' not in s, path
print('OK patch_verbose')
