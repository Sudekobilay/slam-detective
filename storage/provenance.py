"""Versioned source/environment identity, not just a during-run mutation check."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess


def source_hashes(root):
    root = Path(root)
    result = {}
    areas = ('application', 'configuration', 'config', 'domain', 'evaluation', 'storage',
             'simulator', 'estimation', 'perception', 'experiments', 'analytics',
             'visualization', 'dataset_loaders', 'tests')
    candidates = [root / name for name in ('main.py', 'pyproject.toml', 'requirements.txt')]
    for area in areas:
        folder = root / area
        if folder.is_dir():
            candidates.extend(folder.rglob('*'))
    for path in sorted(candidates):
        relative = path.relative_to(root)
        if any(p in ('__pycache__', 'results', 'build', 'dist') for p in relative.parts):
            continue
        if path.is_file() and (path.suffix in ('.py', '.toml') or relative.parts[0] == 'config' or path.name == 'requirements.txt'):
            result[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def environment_identity(root):
    versions = {}
    for name in ('numpy', 'matplotlib', 'pillow'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    git = {'commit': None, 'dirty': None}
    try:
        commit = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=5, check=True)
        status = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'], capture_output=True, text=True, timeout=5, check=True)
        git = {'commit': commit.stdout.strip(), 'dirty': bool(status.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'dependencies': versions, 'git': git}


def protected_hashes(root):
    """Legacy name; now covers all source/config areas, including evaluation/render."""
    return source_hashes(root)
