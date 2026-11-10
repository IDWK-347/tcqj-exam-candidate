"""Own lifecycle around the pinned MASt3R sparse-GA call; FP32 only."""
import contextlib
import copy
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import torch

from .sparse_io import (
    SPARSE_SOURCE, SPARSE_SOURCE_SHA256, SparseSceneData,
    extract_sparse_scene, prepare_vendor_inputs,
)
from .sparse_pixels import SparsePixelCapture

VENDOR_MANIFEST_SHA256 = '140d74898111e78251fa6b8ea0be59431587e46b773cfe4fd6f67ace39ec0579'
IMPORT_FILES = {
    'mast3r.model': 'vendor/mast3r/mast3r/model.py',
    'mast3r.cloud_opt.sparse_ga': SPARSE_SOURCE,
    'dust3r.utils.image': 'vendor/mast3r/dust3r/dust3r/utils/image.py',
}


class BackendError(ValueError):
    """Invalid backend setup, workspace, execution context or required output."""


def _hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def inspect_vendor_sources(project_root):
    """Inspect source bytes only; this does not import a reconstruction module."""
    root = Path(project_root).resolve()
    manifest = root / 'vendor/manifest.json'
    if not manifest.resolve().is_relative_to(root) or not manifest.is_file():
        raise BackendError('Missing vendor manifest or path outside repository')
    if _hash(manifest) != VENDOR_MANIFEST_SHA256:
        raise BackendError('Vendor manifest SHA256 differs from the inspected snapshot')
    files = {}
    for name, relative in IMPORT_FILES.items():
        path = root / relative
        if not path.resolve().is_relative_to(root / 'vendor/mast3r') or not path.is_file():
            raise BackendError('Missing/outside vendor source: ' + relative)
        files[name] = {'file': relative, 'sha256': _hash(path)}
    if files['mast3r.cloud_opt.sparse_ga']['sha256'] != SPARSE_SOURCE_SHA256:
        raise BackendError('Pinned sparse_ga.py SHA256 differs')
    return {'vendor_manifest_sha256': _hash(manifest), 'files': files}


def _import_vendor(project_root):
    root = Path(project_root).resolve()
    vendor = root / 'vendor/mast3r'
    # Reject conflicting cached modules before importing any new backend code.
    for name, relative in IMPORT_FILES.items():
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module, '__file__', '')).resolve() != (root / relative).resolve():
            raise BackendError('Cached backend import is outside the expected vendor file: ' + name)
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    modules, files = {}, {}
    for name, relative in IMPORT_FILES.items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if path != (root / relative).resolve() or not path.is_relative_to(vendor):
            raise BackendError('Backend import is outside the expected vendor file: ' + name)
        modules[name] = module
        files[name] = {'file': relative, 'sha256': _hash(path)}
    return (modules['mast3r.model'].AsymmetricMASt3R,
            modules['mast3r.cloud_opt.sparse_ga'].sparse_global_alignment, files)


@dataclass(frozen=True)
class BackendCapabilities:
    def metadata(self):
        return {'name': 'mast3r_sparse_ga', 'devices': ['cpu', 'cuda'],
            'precision_modes': ['fp32'], 'sparse_world_observations_supported': True,
            'cameras_supported': True, 'dense_points_supported': False,
            'source_pixels_xy_supported': True,
            'dense_depth_supported': False, 'point_confidence_supported': False,
            'pair_matching_metrics_supported': False, 'probe_pair_supported': True,
            'independent_probe_matches_supported': True,
            'probe_validation_scope': 'CPU contracts; real GPU matching acceptance remains separate',
            'actual_model_forward_count_supported': False,
            'validation_scope': 'CPU contracts; actual GPU acceptance is a separate run'}


@dataclass(frozen=True)
class WeightsSpec:
    path: Path
    sha256: str


@dataclass(frozen=True)
class RuntimeSpec:
    device: str
    precision: str


@dataclass(frozen=True)
class BackendFingerprint:
    values: dict

    @property
    def fingerprint_hash(self):
        raw = json.dumps(self.values, sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False).encode('utf-8')
        return hashlib.sha256(raw).hexdigest()

    def metadata(self):
        return {**copy.deepcopy(self.values), 'fingerprint_hash': self.fingerprint_hash}


@dataclass(frozen=True)
class ReconstructionOptions:
    subsample: int
    coarse_lr: float
    coarse_iterations: int
    refine_lr: float
    refine_iterations: int
    matching_conf_thr: float
    shared_intrinsics: bool

    def upstream_kwargs(self, device):
        for name in ('subsample', 'coarse_iterations', 'refine_iterations'):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise BackendError('Positive integer required: ' + name)
        for name in ('coarse_lr', 'refine_lr', 'matching_conf_thr'):
            value = getattr(self, name)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or value < 0 or (name != 'matching_conf_thr' and value == 0)):
                raise BackendError('Invalid finite optimization value: ' + name)
        if type(self.shared_intrinsics) is not bool:
            raise BackendError('shared_intrinsics must be boolean')
        return {'subsample': self.subsample, 'device': device,
            'lr1': self.coarse_lr, 'niter1': self.coarse_iterations,
            'lr2': self.refine_lr, 'niter2': self.refine_iterations,
            'matching_conf_thr': self.matching_conf_thr,
            'shared_intrinsics': self.shared_intrinsics}


@dataclass(frozen=True)
class BackendWorkspace:
    allowed_root: Path
    run_directory: Path
    cache_namespace: str
    event_callback: object = None

    def validate(self):
        allowed, directory = Path(self.allowed_root).resolve(), Path(self.run_directory).resolve()
        if not directory.is_relative_to(allowed) or not directory.is_dir():
            raise BackendError('Backend run directory must exist inside the allowed root')
        if (not isinstance(self.cache_namespace, str)
                or re.fullmatch(r'[A-Za-z0-9_-]{1,80}', self.cache_namespace) is None):
            raise BackendError('Invalid cache namespace')
        if self.event_callback is not None and not callable(self.event_callback):
            raise BackendError('Backend event callback must be callable')
        log = directory / 'reconstruction_optimizer.log'
        if not log.resolve().is_relative_to(directory) or log.exists():
            raise BackendError('Optimizer log must be a new file inside this run')
        return directory, log

    def emit(self, event, **details):
        if self.event_callback is not None:
            self.event_callback({'event': event, **details})


@dataclass(frozen=True)
class BackendResult:
    fingerprint: BackendFingerprint
    scene: SparseSceneData
    input_mapping: dict
    settings: dict
    final_losses: tuple
    reconstruction_seconds: float
    peak_allocated_gib: object
    workspace: dict

    def metadata(self):
        return {'schema_version': 1, 'backend': 'mast3r_sparse_ga', 'status': 'completed',
            'backend_fingerprint': self.fingerprint.metadata(), 'input_mapping': self.input_mapping,
            'settings': self.settings, 'scene': self.scene.metadata(),
            'final_losses': list(self.final_losses),
            'reconstruction_seconds_excluding_model_load': self.reconstruction_seconds,
            'peak_allocated_gib': self.peak_allocated_gib, 'workspace': self.workspace,
            'execution': {'optimization_grad_enabled': True,
                'algorithm_dtype_argument': 'torch.float32',
                'observed_intrinsics_dtype': str(self.scene.K_processed.dtype),
                'observed_pose_dtype': str(self.scene.T_world_from_camera.dtype),
                'observed_point_dtypes': sorted({str(view.points_world.dtype) for view in self.scene.views}),
                'encoder_and_decoder_internal_dtypes': {'status': 'unavailable', 'value': None,
                    'reason': 'No per-operation dtype instrumentation in this stage'},
                'downstream_head_fp32_policy': {'basis': 'inspected_pinned_source',
                    'autocast_enabled': False, 'tokens_converted_to_float': True}},
            'measurement_scope': {'seconds': 'sparse_global_alignment including constructor pixel capture and device synchronization; excludes extraction/export',
                'peak_allocated_gib': 'PyTorch allocator peak reset after model loading; not total VRAM'},
            'geometry_quality_accepted': False}


def _check_context():
    if torch.is_inference_mode_enabled():
        raise BackendError('Geometry optimization cannot run inside inference_mode')
    if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
        raise BackendError('FP32 backend cannot inherit an external autocast context')


class MASt3RSparseBackend:
    def __init__(self, project_root):
        self.root = Path(project_root).resolve()
        self._model = None
        self._alignment = None
        self._fingerprint = None
        self._device = None
        self._requested_device = None

    def describe(self):
        return BackendCapabilities()

    def prepare(self, weights_spec, runtime_spec):
        if self._model is not None:
            raise BackendError('Backend is already prepared; close it before preparing again')
        if not isinstance(weights_spec, WeightsSpec) or not isinstance(runtime_spec, RuntimeSpec):
            raise BackendError('Expected own WeightsSpec and RuntimeSpec')
        _check_context()
        path = Path(weights_spec.path).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise BackendError('Weights must be an existing local file inside the repository')
        if (not isinstance(weights_spec.sha256, str)
                or re.fullmatch(r'[0-9a-f]{64}', weights_spec.sha256) is None
                or _hash(path) != weights_spec.sha256):
            raise BackendError('Checkpoint SHA256 mismatch or invalid declaration')
        if runtime_spec.precision != 'fp32':
            raise BackendError('Backend currently supports only fp32')
        if not isinstance(runtime_spec.device, str):
            raise BackendError('Device must be an explicit string')
        try:
            device = torch.device(runtime_spec.device)
        except (RuntimeError, ValueError) as error:
            raise BackendError('Invalid runtime device') from error
        if device.type not in ('cpu', 'cuda') or (device.type == 'cpu' and device.index is not None):
            raise BackendError('Supported devices are cpu and cuda[:index]')
        if device.type == 'cuda':
            if not torch.cuda.is_available():
                raise BackendError('CUDA is unavailable; no CPU fallback')
            index = device.index if device.index is not None else torch.cuda.current_device()
            if index >= torch.cuda.device_count():
                raise BackendError('CUDA index is outside available devices')
            device = torch.device('cuda', index)
        sources = inspect_vendor_sources(self.root)
        model_class, alignment, loaded_files = _import_vendor(self.root)
        if loaded_files != sources['files']:
            raise BackendError('Imported source bytes differ from inspected sources')
        self._device, self._requested_device = device, runtime_spec.device
        try:
            self._model = model_class.from_pretrained(str(path))
            self._model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
            parameters = list(self._model.parameters())
            floating = [parameter for parameter in parameters if parameter.is_floating_point()]
            buffers = [buffer for buffer in self._model.buffers() if buffer.is_floating_point()]
            if (not floating or any(parameter.requires_grad for parameter in parameters)
                    or any(value.dtype != torch.float32 or value.device != device
                           for value in floating + buffers)):
                raise BackendError('Observed model dtype/device/frozen state differs from FP32 setup')
            self._alignment = alignment
            self._fingerprint = BackendFingerprint({
                'schema_version': 1, 'backend': 'mast3r_sparse_ga', 'source_fingerprint': sources,
                'weight_resource': path.relative_to(self.root).as_posix(), 'weight_sha256': weights_spec.sha256,
                'requested_device': runtime_spec.device, 'requested_precision': runtime_spec.precision,
                'observed_model_parameter_dtypes': sorted({str(value.dtype) for value in floating}),
                'observed_floating_buffer_dtypes': sorted({str(value.dtype) for value in buffers}),
                'observed_model_devices': sorted({str(value.device) for value in floating + buffers}),
                'model_parameters_frozen': True, 'model_eval': not self._model.training,
                'parameter_count': sum(value.numel() for value in parameters),
                'torch_version': str(torch.__version__), 'torch_cuda_version': torch.version.cuda,
                'cuda_device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None})
            return self._fingerprint
        except Exception:
            self.close()
            raise

    def reconstruct(self, images, pair_graph, options, workspace):
        if self._model is None or self._alignment is None:
            raise BackendError('Backend must be prepared before reconstruction')
        if not isinstance(options, ReconstructionOptions) or not isinstance(workspace, BackendWorkspace):
            raise BackendError('Expected own ReconstructionOptions and BackendWorkspace')
        _check_context()
        settings = options.upstream_kwargs(self._requested_device)
        inputs = prepare_vendor_inputs(images, pair_graph)
        directory, log_path = workspace.validate()
        device = self._device
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        log = io.StringIO()
        cache_relative = None
        try:
            with tempfile.TemporaryDirectory(prefix='sparse_cache_' + workspace.cache_namespace + '_',
                                             dir=directory) as cache:
                cache_relative = Path(cache).relative_to(directory).as_posix()
                workspace.emit('alignment_started', cache_directory=cache_relative,
                               directed_item_count=len(inputs.pairs))
                capture = SparsePixelCapture(self._alignment, inputs)
                started = time.perf_counter()
                try:
                    with torch.enable_grad(), contextlib.redirect_stdout(log):
                        scene = capture.alignment(list(inputs.paths), list(inputs.pairs), cache,
                                                self._model, **settings, opt_pp=False, dtype=torch.float32)
                    if device.type == 'cuda':
                        torch.cuda.synchronize(device)
                    elapsed = time.perf_counter() - started
                    peak = (torch.cuda.max_memory_allocated(device) / 1024**3
                            if device.type == 'cuda' else None)
                finally:
                    with log_path.open('x', encoding='utf-8', newline='\n') as stream:
                        stream.write(log.getvalue())
                    print(log.getvalue(), flush=True)
                workspace.emit('alignment_returned', reconstruction_seconds=elapsed)
                losses = tuple(float(value) for value in re.findall(r'>> final loss = (\S+)', log.getvalue()))
                if len(losses) != 2 or not np.isfinite(losses).all():
                    raise BackendError('Optimizer loss check failed; inspect reconstruction_optimizer.log')
                data = extract_sparse_scene(scene, inputs, pixels_xy=capture.pixels_for(scene),
                                            pixel_source=capture.metadata())
                if (data.K_processed.dtype != np.float32 or data.T_world_from_camera.dtype != np.float32
                        or any(view.points_world.dtype != np.float32 for view in data.views)):
                    raise BackendError('Observed optimization output dtype differs from FP32')
                del scene, capture
                workspace.emit('result_extracted', observation_count=data.observation_count,
                               per_image_counts=[len(view.points_world) for view in data.views])
        finally:
            if cache_relative is not None:
                workspace.emit('cache_removed', cache_directory=cache_relative,
                               removed=not (directory / cache_relative).exists())
        return BackendResult(self._fingerprint, data, inputs.metadata(), settings,
            losses, elapsed, peak, {'cache_namespace': workspace.cache_namespace,
                'temporary_cache_directory': cache_relative, 'temporary_cache_cleaned': True,
                'cross_run_cache_reuse': False})

    def probe_pair(self, image_a, image_b, options, workspace):
        from .probe import run_pair_probe
        return run_pair_probe(self, image_a, image_b, options, workspace)

    def close(self):
        device = self._device
        self._model = None
        self._alignment = None
        self._fingerprint = None
        self._device = None
        self._requested_device = None
        if device is not None and device.type == 'cuda':
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
