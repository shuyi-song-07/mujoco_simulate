"""Import/path compatibility for a reviewed directory-only migration.

Aliases return the actual module object (including private attributes), so old
imports, pickle names and test patches do not create a second implementation.
Source compatibility is limited to exact before/after hashes in a checked-in
manifest. A later edit is never covered merely because its path is listed.
"""
from __future__ import annotations
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name('layout_migration.json')
PREFIXES = {'dataset.task3': 'dataset.common', 'training.papera': 'training.act'}
MODULES = {f'rollout.{name}': f'rollout.common.{name}' for name in
           ('action_chunks', 'diffusion_policy', 'executor', 'policy', 'protocol', 'rollout')}
MODULES.update({f'simulation.mujoco.dual_arm.{name}': f'simulation.mujoco.common.{name}' for name in
               ('command_protocol','hand_measurement','measurement_journal','recording_policy',
                'recording_timeline','render_workers','session_diagnostics','teleop_service',
                'service_identity','training_camera','render_identity')})
MODULES.update({f'tasks.{name}.evidence': f'dataset.{name}.evidence' for name in
               ('push_t','h_frame_push_align','push_r')})


def canonical_module(name):
    if name in MODULES: return MODULES[name]
    for old, new in PREFIXES.items():
        if name == old or name.startswith(old + '.'): return new + name[len(old):]
    return name


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, old, new): self.old, self.new = old, new
    def create_module(self, spec):
        module = importlib.import_module(self.new)
        self.actual_spec = module.__spec__
        return module
    def exec_module(self, module):
        # importlib temporarily assigns the alias spec to this shared object.
        # Restore the real loader so later reload/runpy cannot recurse via alias.
        module.__spec__ = self.actual_spec
        module.__loader__ = self.actual_spec.loader
    def get_code(self, fullname):
        return importlib.util.find_spec(self.new).loader.get_code(self.new)
    def get_source(self, fullname): return None
    def is_package(self, fullname):
        spec = importlib.util.find_spec(self.new)
        return bool(spec and spec.submodule_search_locations is not None)


class _CanonicalLoader(importlib.abc.Loader):
    """Delegate actual loading; tolerate an old name requested by runpy."""
    def __init__(self, name, loader): self.name, self.loader = name, loader
    def create_module(self, spec): return self.loader.create_module(spec)
    def exec_module(self, module): return self.loader.exec_module(module)
    def get_code(self, fullname): return self.loader.get_code(self.name)
    def get_source(self, fullname): return self.loader.get_source(self.name)
    def get_filename(self, fullname): return self.loader.get_filename(self.name)
    def is_package(self, fullname): return self.loader.is_package(self.name)
    def __getattr__(self, name): return getattr(self.loader, name)


class _AliasFinder(importlib.abc.MetaPathFinder):
    _papera_layout_alias = True
    def find_spec(self, fullname, path=None, target=None):
        new = canonical_module(fullname)
        if new == fullname:
            canonical = fullname in MODULES.values() or any(
                fullname == prefix or fullname.startswith(prefix + '.') for prefix in PREFIXES.values())
            if not canonical: return None
            actual = importlib.machinery.PathFinder.find_spec(fullname, path)
            if actual and actual.loader and hasattr(actual.loader, 'get_code'):
                actual.loader = _CanonicalLoader(fullname, actual.loader)
            return actual
        actual = importlib.util.find_spec(new)
        if actual is None: return None
        spec = importlib.util.spec_from_loader(fullname, _AliasLoader(fullname, new),
                                              is_package=actual.submodule_search_locations is not None)
        spec.origin = actual.origin
        return spec


def install_aliases():
    if not any(getattr(f, '_papera_layout_alias', False) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())


def _manifest():
    return json.loads(MANIFEST.read_text(encoding='utf-8')) if MANIFEST.is_file() else {'files': {}}


def _relative(path):
    p = Path(path)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    return p, p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else None


def source_path(path):
    """Resolve legacy project source names, never redirect data outside ROOT."""
    p, relative = _relative(path)
    if relative is None: return p
    record = _manifest()['files'].get(relative)
    if record: return ROOT / record['path']
    # Prefixes also support directory glob callers before/after migration.
    for old, new in PREFIXES.items():
        a, b = old.replace('.', '/'), new.replace('.', '/')
        if relative == a or relative.startswith(a + '/'):
            candidate = ROOT / (b + relative[len(a):])
            if candidate.exists(): return candidate
    module = relative[:-3].replace('/', '.') if relative.endswith('.py') else None
    if module in MODULES:
        candidate = ROOT / (MODULES[module].replace('.', '/') + '.py')
        if candidate.is_file(): return candidate
    return p


def logical_source_path(path):
    """Return the historical source identifier as an absolute Path."""
    p, relative = _relative(path)
    for old, record in _manifest()['files'].items():
        if record['path'] == relative: return ROOT / old
    return p


def migration_reason(path, old_hash, new_hash):
    p, relative = _relative(path)
    if relative is None: return None
    manifest = _manifest()
    for old, record in manifest['files'].items():
        if relative in (old, record['path']) and old_hash == record.get('before_sha256') and new_hash == record.get('after_sha256'):
            actual = source_path(p)
            if (actual.is_file() and hashlib.sha256(actual.read_bytes()).hexdigest() == new_hash
                    and all((ROOT/name).is_file() and hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
                            for name, digest in manifest.get('bundle', {}).items())):
                return 'task-directory-relocation-20261113'
    return None


def source_hash_matches(path, expected):
    actual_path = source_path(path)
    if not actual_path.is_file(): return False
    actual = hashlib.sha256(actual_path.read_bytes()).hexdigest()
    return actual == expected or migration_reason(path, expected, actual) is not None


def equivalent_binding(recorded, current):
    """Only certified source moves may differ; scene/config/manifest stay exact."""
    if recorded == current: return True
    if not isinstance(recorded, dict) or not isinstance(current, dict): return False
    before, after = dict(recorded), dict(current)
    a, b = before.pop('sources', None), after.pop('sources', None)
    if before != after or not isinstance(a, dict) or not isinstance(b, dict): return False
    if a.keys() != b.keys(): return source_mapping_matches(a, b)
    return all(a[key] == b[key] or migration_reason(ROOT/key, a[key], b[key]) for key in a)


def source_mapping_matches(expected, current):
    """Match full source catalogs, including every newly split dependency.

    Historical catalogs must match an entire audited before/after pair.
    An empty or truncated catalog is not evidence of compatible source code.
    """
    if not isinstance(expected, dict) or not isinstance(current, dict): return False
    if expected == current: return bool(expected)
    manifest = _manifest()
    pair = next((pair for pair in manifest.get('catalogs', {}).values()
                 if expected == pair['before'] and current == pair['after']), None)
    if pair is None: return False
    return all(source_path(ROOT/name).is_file() and hashlib.sha256(source_path(ROOT/name).read_bytes()).hexdigest() == digest
               for name, digest in {**manifest.get('bundle', {}), **current}.items())
