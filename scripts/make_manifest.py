"""Explicitly record the inputs for a new reproduction-code revision."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    files = []
    for folder in ['models', 'envs', 'utils', 'data', 'scripts', 'followup', 'tests',
                   'examples', 'experiments', 'requirements', 'reproduction/analysis',
                   'reproduction/reference', 'reproduction/environments']:
        files.extend(p for p in (ROOT / folder).rglob('*')
                     if p.is_file() and p.suffix in ['.py', '.json', '.tex', '.txt']
                     and '__pycache__' not in p.parts)
    files.extend(ROOT / name for name in ['reproduce.py', 'requirements.txt', 'LICENSE',
                                         'followup/vendor/diffusion_policy/LICENSE',
                                         'reproduction/EXPECTED_NUMERICS.json'])
    record = {'scope': 'Executable code, configs, archived reports, numerical references and dependency records.',
              'files': {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted(set(files))}}
    (ROOT / 'reproduction/MANIFEST.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f"Recorded {len(record['files'])} inputs. Re-run the checks before publishing this revision.")


if __name__ == '__main__':
    main()
