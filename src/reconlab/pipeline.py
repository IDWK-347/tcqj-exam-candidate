"""Pipeline stage/resource lifecycle; no model, tensor or global grad context.

This module records a caller's explicit stage plan. ReconstructionPipeline,
exported below, binds the normal CLI to the owned reconstruction modules.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid


class PipelineError(ValueError):
    """Invalid phase transition or incomplete run."""


@dataclass(frozen=True)
class RunManifest:
    values: dict

    def metadata(self): return deepcopy(self.values)


def _utc(): return datetime.now(timezone.utc).isoformat()


def _require(condition, message):
    if not condition: raise PipelineError(message)


def _serializable(value):
    try:
        encoded = json.dumps(value, allow_nan=False)
        _require(json.loads(encoded) == value, 'Metadata must round-trip as JSON without changing types/keys')
    except (ValueError, TypeError) as error:
        raise PipelineError(f'Metadata must be finite JSON data: {error}') from error


def _error(error, phase):
    return {'phase': phase, 'type': type(error).__name__, 'message': str(error)}


class PipelineRun:
    """Fresh run directory, ordered stages, terminal manifest and owned cleanup.

Use as a context manager, with one stage() context for each declared phase.
Callers register owned resource closers immediately after acquiring resources.
Cleanup runs in reverse acquisition order and never hides the primary error.
"""
    def __init__(self, output_root, scene_id, phase_names, *, mode, provenance):
        _require(isinstance(scene_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', scene_id),
                 'Scene ID must be a safe directory component')
        _require(isinstance(phase_names, (tuple, list)) and len(phase_names) > 0
                 and all(isinstance(name, str) and re.fullmatch(r'[a-z][a-z0-9_]*', name) for name in phase_names)
                 and len(set(phase_names)) == len(phase_names), 'Invalid unique phase names')
        _require(mode in ('reconstruction', 'cpu_export_replay'), 'Unknown run mode')
        _require(isinstance(provenance, dict), 'Provenance must be a mapping')
        _serializable(provenance)
        self.run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:12]
        root = Path(output_root).resolve()
        self.run_dir = root / scene_id / self.run_id
        _require(self.run_dir.resolve().is_relative_to(root), 'Run directory outside output root')
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self._data = {
            'schema_version': 1, 'run_id': self.run_id, 'scene_id': scene_id, 'mode': mode,
            'status': 'pending', 'created_at_utc': _utc(), 'started_at_utc': None, 'ended_at_utc': None,
            'provenance': deepcopy(provenance), 'phases': [
                {'name': name, 'status': 'pending', 'started_at_utc': None, 'ended_at_utc': None, 'error': None}
                for name in phase_names],
            'resources': [], 'cleanup_errors': [], 'persistence_errors': [], 'error': None, 'artifacts': None,
            'geometry_quality_accepted': False,
        }
        self._active = None
        self._failure = None
        self._failure_phase = None
        self._callbacks = []
        self._cleanup_failures = []
        self._closing = False
        self._checkpoint()

    @property
    def manifest(self): return RunManifest(deepcopy(self._data))

    def _checkpoint(self):
        """Each manifest/status file is atomically replaced inside this fresh run."""
        status = {key: self._data[key] for key in ('schema_version', 'run_id', 'status', 'error', 'geometry_quality_accepted')}
        for name, value in (('run_manifest.json', self._data), ('run_status.json', status)):
            payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
            temporary = self.run_dir / ('.manifest-' + uuid.uuid4().hex + '.tmp')
            try:
                with temporary.open('xb') as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.run_dir / name)
            finally:
                if temporary.exists(): temporary.unlink()

    def __enter__(self):
        _require(self._data['status'] == 'pending', 'The same run cannot be entered twice')
        self._data['status'] = 'running'
        self._data['started_at_utc'] = _utc()
        try: self._checkpoint()
        except BaseException as error:
            self._data['status'] = 'failed'
            self._data['error'] = _error(error, 'pipeline_start')
            self._data['ended_at_utc'] = _utc()
            try: self._checkpoint()
            except BaseException as persistence:
                self._data['persistence_errors'].append(_error(persistence, 'pipeline_start'))
            raise
        return self

    @contextmanager
    def stage(self, name):
        _require(self._data['status'] == 'running' and not self._closing, 'Stages require an active run')
        _require(self._active is None, 'Nested stages are not permitted')
        _require(self._failure is None, 'A failed run cannot advance to another stage')
        pending = [phase for phase in self._data['phases'] if phase['status'] == 'pending']
        _require(pending and pending[0]['name'] == name, 'Stage must follow the declared order exactly once')
        phase = pending[0]
        phase['status'] = 'running'
        phase['started_at_utc'] = _utc()
        self._active = phase
        try:
            self._checkpoint()
            yield
        except BaseException as error:
            phase['status'] = 'failed'
            phase['error'] = _error(error, name)
            self._failure, self._failure_phase = error, name
            raise
        else:
            phase['status'] = 'succeeded'
        finally:
            phase['ended_at_utc'] = _utc()
            self._active = None
            try: self._checkpoint()
            except BaseException as error:
                self._data['persistence_errors'].append(_error(error, name))
                if self._failure is None:
                    phase['status'] = 'failed'
                    phase['error'] = _error(error, name)
                    self._failure, self._failure_phase = error, name
                    raise

    def set_artifacts(self, metadata):
        _require(self._data['status'] == 'running' and self._active is not None,
                 'Artifacts must be recorded inside an active stage')
        _require(isinstance(metadata, dict), 'Artifact metadata must be a mapping')
        _serializable(metadata)
        self._data['artifacts'] = deepcopy(metadata)
        self._checkpoint()

    def update_provenance(self, metadata):
        _require(self._data['status'] == 'running' and self._active is not None,
                 'Provenance updates require an active stage')
        _require(isinstance(metadata, dict), 'Provenance must be a mapping')
        _serializable(metadata)
        self._data['provenance'].update(deepcopy(metadata))
        self._checkpoint()

    def register_resource(self, name, close):
        _require(self._data['status'] == 'running' and not self._closing, 'Resource registration needs an active run')
        _require(isinstance(name, str) and bool(name.strip()) and callable(close)
                 and name not in [item['name'] for item in self._data['resources']], 'Invalid unique resource closer')
        index = len(self._data['resources'])
        self._data['resources'].append({'name': name, 'status': 'open', 'registered_at_utc': _utc(),
            'close_started_at_utc': None, 'closed_at_utc': None, 'error': None})
        self._callbacks.append((index, close))
        self._checkpoint()

    def close_resource(self, name):
        """Release an owned resource now; never repeat a failed or completed close."""
        _require(self._data['status'] == 'running', 'Resource close requires an active run')
        matches = [(position, index, close) for position, (index, close) in enumerate(self._callbacks)
                   if self._data['resources'][index]['name'] == name]
        _require(len(matches) == 1, 'Resource is unknown or already closed')
        position, index, close = matches[0]
        self._callbacks.pop(position)
        resource = self._data['resources'][index]
        resource['close_started_at_utc'] = _utc()
        primary = None
        try:
            close()
        except BaseException as error:
            primary = error
            resource['status'] = 'failed'
            resource['error'] = _error(error, 'resource_cleanup')
            self._data['cleanup_errors'].append({'resource': name, **resource['error']})
            self._cleanup_failures.append(error)
            raise
        else:
            resource['status'] = 'closed'
        finally:
            resource['closed_at_utc'] = _utc()
            try:
                self._checkpoint()
            except BaseException as error:
                self._data['persistence_errors'].append(_error(error, 'resource_cleanup'))
                if primary is None:
                    self._cleanup_failures.append(error)
                    raise

    def _close_resources(self):
        self._closing = True
        for index, _ in list(reversed(self._callbacks)):
            try:
                self.close_resource(self._data['resources'][index]['name'])
            except BaseException:
                pass  # close_resource retains each error; never mask the business error.
        return self._cleanup_failures

    def __exit__(self, error_type, error, traceback):
        primary = error if error is not None else self._failure
        phase = self._failure_phase if primary is self._failure else 'pipeline'
        if primary is None and not all(item['status'] == 'succeeded' for item in self._data['phases']):
            primary = PipelineError('Declared phases did not all succeed')
            phase = 'pipeline_completion'
        cleanup = self._close_resources()
        if primary is None and cleanup:
            primary, phase = cleanup[0], 'resource_cleanup'
        self._data['error'] = None if primary is None else _error(primary, phase)
        self._data['status'] = 'succeeded' if primary is None else 'failed'
        self._data['ended_at_utc'] = _utc()
        try: self._checkpoint()
        except BaseException as persistence:
            self._data['persistence_errors'].append(_error(persistence, 'final_manifest'))
            if primary is None: raise
        if error is None and primary is not None: raise primary
        return False


from .reconstruction_pipeline import ReconstructionPipeline
