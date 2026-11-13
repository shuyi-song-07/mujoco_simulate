"""Evaluation ownership remains exclusive on Windows and POSIX servers."""
import builtins
import errno
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from rollout import rollout

ROOT = Path(__file__).resolve().parents[2]


class PosixFileLocks(ModuleType):
    """Simulate the unavailable OS boundary when running these tests on Windows."""
    LOCK_EX = 2
    LOCK_NB = 4
    LOCK_UN = 8

    def __init__(self):
        super().__init__('fcntl')
        self.owners = set()

    def flock(self, descriptor, operation):
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino)
        if operation == self.LOCK_EX | self.LOCK_NB:
            if identity in self.owners:
                raise BlockingIOError(errno.EAGAIN, 'Resource temporarily unavailable')
            self.owners.add(identity)
        elif operation == self.LOCK_UN:
            self.owners.remove(identity)
        else:
            raise AssertionError('Evaluation ownership must use a nonblocking exclusive lock')


class EvaluationLockTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / '.local/tmp'
        parent.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix='rollout_lock_', dir=parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.root_patch = patch.object(rollout, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def probe_other_process(self):
        code = """
from pathlib import Path
import sys
from rollout import rollout
rollout.ROOT = Path(sys.argv[1])
try:
    with rollout.evaluation_lock():
        print('acquired')
except RuntimeError as error:
    if 'Another evaluation is running' not in str(error):
        raise
    print('blocked')
    raise SystemExit(23)
"""
        return subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', code, str(self.root)],
                              cwd=ROOT, capture_output=True, text=True, timeout=20)

    def test_native_lock_blocks_other_process_and_releases_normally(self):
        with rollout.evaluation_lock():
            blocked = self.probe_other_process()
            self.assertEqual((blocked.returncode, blocked.stdout.strip()),
                             (23, 'blocked'), blocked.stderr)
        available = self.probe_other_process()
        self.assertEqual((available.returncode, available.stdout.strip()),
                         (0, 'acquired'), available.stderr)

    def test_native_lock_releases_after_evaluation_error(self):
        with self.assertRaisesRegex(ValueError, 'evaluation body failed'):
            with rollout.evaluation_lock():
                raise ValueError('evaluation body failed')
        available = self.probe_other_process()
        self.assertEqual((available.returncode, available.stdout.strip()),
                         (0, 'acquired'), available.stderr)

    def test_posix_without_windows_module_keeps_ownership_until_context_exit(self):
        backend = PosixFileLocks()
        real_import = builtins.__import__

        def server_import(name, *args, **kwargs):
            if name == 'msvcrt':
                raise ModuleNotFoundError("No module named 'msvcrt'")
            return real_import(name, *args, **kwargs)

        with patch.object(rollout, 'os', SimpleNamespace(name='posix')), \
                patch.dict(sys.modules, {'fcntl': backend}), \
                patch('builtins.__import__', side_effect=server_import):
            with self.assertRaisesRegex(ValueError, 'evaluation body failed'):
                with rollout.evaluation_lock():
                    with self.assertRaisesRegex(RuntimeError, 'Another evaluation is running'):
                        with rollout.evaluation_lock():
                            self.fail('A second evaluation acquired an occupied lock')
                    raise ValueError('evaluation body failed')
            with rollout.evaluation_lock():
                pass
        self.assertFalse(backend.owners)


if __name__ == '__main__':
    unittest.main()
