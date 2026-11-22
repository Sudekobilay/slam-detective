"""Controller history retention, separate from estimator state."""
from collections import deque
import json
from pathlib import Path


class MemoryRecorder:
    def __init__(self):
        self.streams = {}

    def stream(self, name):
        return self.streams.setdefault(name, [])


class JsonlStream(deque):
    def __init__(self, path, maxlen):
        super().__init__(maxlen=maxlen)
        self.path = path

    def append(self, value):
        line = json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(line)
        super().append(value)


class JsonlRecorder:
    """Append-only full history on disk, bounded recent controller history in RAM.

    Directory must be new; no blind append to an earlier run. A crash may leave
    a partial final JSONL line; this recorder does not claim automatic resume.
    Simulator/backend/guard histories are separate and remain experiment-sized.
    """
    def __init__(self, directory, *, recent_rows=100):
        if type(recent_rows) is not int or recent_rows < 1:
            raise ValueError('Positive history bound required.')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.recent_rows = recent_rows
        self.streams = {}

    def stream(self, name):
        if name not in ('path', 'reference_path', 'history', 'raw_history'):
            raise ValueError('Unknown controller history.')
        if name not in self.streams:
            self.streams[name] = JsonlStream(self.directory / (name + '.jsonl'), self.recent_rows)
        return self.streams[name]
