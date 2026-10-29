"""Verify the reproducibility inputs and run paper regeneration or miniature checks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / 'reproduction'


def run(*args):
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def verify():
    manifest = json.loads((PAPER / 'MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Reproducibility input changed: {name}')
    print(f"Verified {len(manifest['files'])} reproducibility inputs.", flush=True)


def figures():
    (PAPER / 'generated').mkdir(parents=True, exist_ok=True)
    expected = json.loads((PAPER / 'EXPECTED_NUMERICS.json').read_text())
    for name in expected:
        (PAPER / 'generated' / name).unlink(missing_ok=True)
    for script in ['matched_study', 'followups', 'primary_sensitivity', 'factorial_figure',
                   'contact_figure', 'visual_narrative', 'architecture_figure', 'task_figure']:
        run(str(PAPER / 'analysis' / (script + '.py')))
    for name, expected_hash in expected.items():
        if hashlib.sha256((PAPER / 'generated' / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f'Numerical output differs: {name}')
    print(f'Regenerated eight figures and matched {len(expected)} numerical outputs exactly.', flush=True)


def execute(stage):
    if stage == 'figures':
        figures()
    elif stage in ['trace', 'sensitivity']:
        script = 'decision_trace' if stage == 'trace' else 'primary_sensitivity'
        run(str(PAPER / 'analysis' / (script + '.py')))
    elif stage == 'smoke':
        run('-m', 'unittest', 'tests.test_tmlr_factorial', '-q')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify', 'figures', 'trace', 'sensitivity', 'smoke', 'check'])
    args = parser.parse_args()
    begin = time.monotonic()
    verify()
    for stage in ['figures', 'trace', 'sensitivity', 'smoke'] if args.stage == 'check' else [args.stage]:
        execute(stage)
    print(f'{args.stage} completed in {time.monotonic()-begin:.1f} seconds.', flush=True)


if __name__ == '__main__':
    main()
