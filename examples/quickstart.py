"""Collect, train and evaluate a tiny point-mass WAM on a CPU; not paper evidence."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(module, *args):
    subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/quickstart')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    data, checkpoint = out / 'toy.npz', out / 'wam.pt'
    run('scripts.collect_data', '--env', 'toy', '--episodes', 12, '--max-steps', 20,
        '--seed', 42, '--output-path', data)
    run('scripts.train', '--dataset-path', data, '--save-path', checkpoint,
        '--model_type', 'wam', '--predict_state', '--epochs', 2, '--horizon', 3,
        '--batch-size', 16, '--action-chunk', 2, '--d-model', 32,
        '--policy-hidden', 64, '--policy-layers', 2, '--device', 'cpu')
    run('scripts.evaluate', '--checkpoint', checkpoint, '--max-horizon', 3,
        '--device', 'cpu', '--output', out / 'prediction.json')
    run('scripts.test', '--checkpoint', checkpoint, '--episodes', 2,
        '--max-steps', 20, '--device', 'cpu', '--json-out', out / 'control.json')
    print(f'Example data, weights and diagnostics: {out}')
    print('Two epochs exercise the pipeline; they are not a trained benchmark result.')


if __name__ == '__main__':
    main()
