"""Explicit run lifecycle; completed cases remain inspectable after interruption."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from storage.files import write_json
from storage.provenance import environment_identity, source_hashes


class RunSession:
    def __init__(self, folder, root, config, *, experiment_id):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        serialized = json.dumps(config, ensure_ascii=False, sort_keys=True, allow_nan=False).encode('utf-8')
        self.document = {'schema_version': 1, 'run_id': self.folder.name,
                         'experiment_id': experiment_id, 'status': 'RUNNING',
                         'started_utc': datetime.now(timezone.utc).isoformat(),
                         'environment': environment_identity(root), 'config': config,
                         'config_sha256': hashlib.sha256(serialized).hexdigest(),
                         'source_sha256': source_hashes(root), 'completed_cases': 0}
        self.finished = False

    def __enter__(self):
        self._write()
        return self

    def _write(self):
        write_json(self.folder / 'run_manifest.json', self.document)

    def checkpoint(self, completed_cases):
        self.document['completed_cases'] = completed_cases
        self._write()

    def finish(self, *, complete):
        self.document['status'] = 'COMPLETED' if complete else 'PARTIAL'
        self.document['ended_utc'] = datetime.now(timezone.utc).isoformat()
        self._write()
        self.finished = True

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            if self.finished and self.document['status'] == 'PARTIAL':
                self.document['error'] = f'{kind.__name__}: {error}'
                self._write()
                return False
            self.document['status'] = 'PARTIAL' if issubclass(kind, KeyboardInterrupt) else 'FAILED'
            self.document['error'] = f'{kind.__name__}: {error}'
        elif not self.finished:
            self.document['status'] = 'FAILED'
            self.document['error'] = 'Runner exited without explicitly finishing.'
        self.document['ended_utc'] = datetime.now(timezone.utc).isoformat()
        self._write()
        return False


def validate_run(folder, *, experiment_id, allow_partial=False, require_manifest=False):
    """Read contract for consumers; legacy archived runs need not have a manifest."""
    from storage.files import read_json
    folder = Path(folder)
    summary = read_json(folder / 'summary.json')
    if summary.get('experiment_id') != experiment_id:
        raise ValueError('Wrong experiment result schema.')
    manifest = folder / 'run_manifest.json'
    if manifest.exists():
        document = read_json(manifest)
        allowed = ('COMPLETED', 'PARTIAL') if allow_partial else ('COMPLETED',)
        if document.get('schema_version') != 1 or document.get('status') not in allowed:
            raise ValueError('Run is not complete; partial results require explicit opt-in.')
    elif require_manifest:
        raise ValueError('Run manifest required.')
    if not allow_partial and not summary.get('all_horizons_completed'):
        raise ValueError('Required horizons are incomplete.')
    return summary
