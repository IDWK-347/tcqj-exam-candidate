"""Independent matching observations over pinned MASt3R functions.

No scene optimization, private cache reading, or second image preprocessing.
CPU-owned arrays retain all merged reciprocal matches and a separate filter mask.
"""
from dataclasses import dataclass, replace
import importlib
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from ..image_loader import ImageRecord
from ..pairing import build_pair_graph
from .sparse_io import prepare_vendor_inputs
from .mast3r import (
    BackendError, BackendWorkspace, IMPORT_FILES, _check_context, _hash,
    inspect_vendor_sources,
)

FAST_NN_SOURCE = 'vendor/mast3r/mast3r/fast_nn.py'
FAST_NN_SHA256 = '2997b5041853b9ff8d072ef2c0f533a30630cd1ef960c1a3f5b293e6f6870d90'


@dataclass(frozen=True)
class ProbeOptions:
    subsample: int
    min_descriptor_confidence: object = None
    border_pixels: int = 0

    def metadata(self):
        if type(self.subsample) is not int or self.subsample <= 0:
            raise BackendError('Probe subsample must be a positive integer')
        threshold = self.min_descriptor_confidence
        if threshold is not None and (type(threshold) not in (int, float)
                or not math.isfinite(threshold) or threshold < 0):
            raise BackendError('Probe descriptor confidence threshold must be finite and nonnegative or None')
        if type(self.border_pixels) is not int or self.border_pixels < 0:
            raise BackendError('Probe border must be a nonnegative integer')
        return {'subsample': self.subsample, 'min_descriptor_confidence': threshold,
                'border_pixels': self.border_pixels, 'threshold_comparison': '>=',
                'confidence_filter_scope': 'merged reciprocal matches; separate from optimizer threshold'}


def prepare_probe_inputs(image_a, image_b):
    if not isinstance(image_a, ImageRecord) or not isinstance(image_b, ImageRecord):
        raise BackendError('Probe requires two own ImageRecords')
    for record in (image_a, image_b):
        if type(record.manifest_index) is not int or record.manifest_index < 0:
            raise BackendError('Invalid original manifest index')
    # Renumber only temporary records, preserving caller identity and tensors.
    batch = SimpleNamespace(records=[replace(image_a, manifest_index=0),
                                     replace(image_b, manifest_index=1)])
    graph = build_pair_graph(batch, {'strategy': 'complete'})
    return prepare_vendor_inputs(batch, graph), graph.edges[0].pair_id


def inspect_probe_sources(project_root):
    sources = inspect_vendor_sources(project_root)
    root = Path(project_root).resolve()
    path = root / FAST_NN_SOURCE
    if not path.resolve().is_relative_to(root / 'vendor/mast3r') or not path.is_file():
        raise BackendError('Missing/outside pinned fast_nn.py')
    if _hash(path) != FAST_NN_SHA256:
        raise BackendError('Pinned fast_nn.py SHA256 differs')
    sources['files']['mast3r.fast_nn'] = {'file': FAST_NN_SOURCE, 'sha256': _hash(path)}
    return sources


def import_probe_functions(project_root, prepared_sources):
    sources = inspect_probe_sources(project_root)
    base = {**sources, 'files': {name: sources['files'][name] for name in IMPORT_FILES}}
    if base != prepared_sources:
        raise BackendError('Probe sources differ from the prepared model fingerprint')
    root = Path(project_root).resolve()
    paths = {**IMPORT_FILES, 'mast3r.fast_nn': FAST_NN_SOURCE}
    # prepare already established the vendor import path; check cached identities
    # before importing the matching functions, then check actual module files.
    for name, relative in paths.items():
        module = sys.modules.get(name)
        if module is not None and Path(getattr(module, '__file__', '')).resolve() != (root / relative).resolve():
            raise BackendError('Cached probe module is outside expected vendor file: ' + name)
    modules = {}
    for name in ('mast3r.cloud_opt.sparse_ga', 'mast3r.fast_nn'):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (root / paths[name]).resolve():
            raise BackendError('Probe import is outside expected vendor file: ' + name)
        if _hash(root / paths[name]) != sources['files'][name]['sha256']:
            raise BackendError('Probe source changed during import: ' + name)
        modules[name] = module
    sparse, fast = modules['mast3r.cloud_opt.sparse_ga'], modules['mast3r.fast_nn']
    if sparse.fast_reciprocal_NNs is not fast.fast_reciprocal_NNs or sparse.merge_corres is not fast.merge_corres:
        raise BackendError('Sparse matching bindings differ from pinned fast_nn module')
    for function in (sparse.symmetric_inference, sparse.extract_correspondences):
        if not callable(function) or function.__module__ != sparse.__name__:
            raise BackendError('Probe function does not belong to pinned sparse module')
    return sparse.symmetric_inference, sparse.extract_correspondences, sources


def validate_probe_predictions(predictions, image_a, image_b):
    if not isinstance(predictions, (tuple, list)) or len(predictions) != 4:
        raise BackendError('Expected four symmetric prediction dictionaries in a,b,b,a order')
    descs, confidences = [], []
    for prediction, record in zip(predictions, (image_a, image_b, image_b, image_a)):
        if not isinstance(prediction, dict):
            raise BackendError('Probe prediction must be a dictionary')
        desc, conf = prediction.get('desc'), prediction.get('desc_conf')
        hw = record.processed_hw
        if (not isinstance(desc, torch.Tensor) or desc.ndim != 4
                or tuple(desc.shape[:3]) != (1, *hw) or desc.shape[3] <= 0
                or desc.dtype != torch.float32
                or not isinstance(conf, torch.Tensor) or tuple(conf.shape) != (1, *hw)
                or conf.dtype != torch.float32 or not bool(torch.isfinite(desc).all())
                or not bool(torch.isfinite(conf).all()) or not bool((conf >= 0).all())):
            raise BackendError('Invalid probe descriptor/confidence shape, FP32 dtype or finite values')
        descs.append(desc[0])
        confidences.append(conf[0])
    if len({desc.shape[-1] for desc in descs}) != 1:
        raise BackendError('Probe descriptor dimensions differ')
    return descs, confidences


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.array(value, copy=True)


def extract_probe_matches(matches, image_a, image_b, options):
    options.metadata()
    if not isinstance(matches, (tuple, list)) or len(matches) != 3:
        raise BackendError('Expected probe xy_a, xy_b, descriptor confidence')
    xy_a, xy_b, confidence = (_cpu_copy(value) for value in matches)
    count = len(confidence) if confidence.ndim == 1 else -1
    for xy, record in ((xy_a, image_a), (xy_b, image_b)):
        h, w = record.processed_hw
        if (xy.shape != (count, 2) or xy.dtype.kind not in 'iu'
                or np.any(xy < 0) or np.any(xy[:, 0] >= w) or np.any(xy[:, 1] >= h)):
            raise BackendError('Probe matches must be integer processed x,y coordinates inside each image')
    if (confidence.dtype != np.float32 or confidence.ndim != 1
            or not np.isfinite(confidence).all() or np.any(confidence < 0)):
        raise BackendError('Probe descriptor confidence must be finite nonnegative FP32 [M]')
    if len(np.unique(np.column_stack((xy_a, xy_b)), axis=0)) != count:
        raise BackendError('Pinned merged reciprocal correspondences must be unique')
    keep = np.ones(count, dtype=np.bool_)
    if options.min_descriptor_confidence is not None:
        keep &= confidence >= options.min_descriptor_confidence
    border = options.border_pixels
    for xy, record in ((xy_a, image_a), (xy_b, image_b)):
        h, w = record.processed_hw
        keep &= (xy[:, 0] >= border) & (xy[:, 0] < w - border)
        keep &= (xy[:, 1] >= border) & (xy[:, 1] < h - border)
    return xy_a, xy_b, confidence, keep


@dataclass(frozen=True)
class PairObservation:
    pair_id: str
    image_a: dict
    image_b: dict
    xy_a: np.ndarray
    xy_b: np.ndarray
    descriptor_confidence: np.ndarray
    keep_mask: np.ndarray
    options: dict
    fingerprint: object
    probe_sources: dict
    inference_seconds: float
    matching_seconds: float
    peak_allocated_gib: object
    workspace: dict

    def metadata(self):
        confidence = self.descriptor_confidence
        stats = {name: float(fn(confidence)) if len(confidence) else None
                 for name, fn in (('min', np.min), ('max', np.max), ('mean', np.mean))}
        return {'schema_version': 1, 'pair_id': self.pair_id,
            'source': 'independent_pair_probe', 'included_in_reconstruction': False,
            'status': 'completed' if self.keep_mask.any() else 'completed_empty',
            'image_a': self.image_a, 'image_b': self.image_b,
            'coordinates': {'space': 'processed_image', 'components': 'x,y',
                'dimensions': 'H,W', 'pixel_centers': 'integer',
                'array_fields': ['xy_a', 'xy_b', 'descriptor_confidence', 'keep_mask']},
            'matching': {'raw_unique_reciprocal_count': len(confidence),
                'retained_count': int(self.keep_mask.sum()),
                'raw_count_scope': 'after upstream reciprocal convergence and direction merging; before own filters',
                'confidence_type': 'geometric_mean_of_endpoint_desc_conf',
                'confidence_formula': 'sqrt(desc_conf_a * desc_conf_b) after direction merge',
                'confidence_is_probability': False, 'confidence_statistics': stats,
                'candidate_count_before_reciprocal_matching': {'status': 'unavailable', 'value': None},
                'geometric_inlier_count': {'status': 'unavailable', 'value': None}},
            'options': self.options, 'backend_fingerprint': self.fingerprint.metadata(),
            'probe_source_fingerprint': self.probe_sources,
            'inference_seconds': self.inference_seconds, 'matching_seconds': self.matching_seconds,
            'peak_allocated_gib': self.peak_allocated_gib,
            'measurement_scope': {'seconds': 'each upstream call plus device synchronization; excludes model load, validation, CPU copies and output files',
                'peak_allocated_gib': 'PyTorch allocator peak reset before probe; not total VRAM'},
            'execution': {'grad_enabled_during_probe': False, 'observed_descriptor_dtype': 'torch.float32',
                'observed_matching_confidence_dtype': str(confidence.dtype),
                'symmetric_inference_call_count': 1,
                'actual_model_forward_count': {'status': 'unavailable', 'value': None}},
            'workspace': self.workspace, 'geometry_quality_accepted': False}


def run_pair_probe(backend, image_a, image_b, options, workspace):
    if backend._model is None or backend._fingerprint is None:
        raise BackendError('Backend must be prepared before an independent probe')
    if not isinstance(options, ProbeOptions) or not isinstance(workspace, BackendWorkspace):
        raise BackendError('Expected own ProbeOptions and BackendWorkspace')
    _check_context()
    option_values = options.metadata()
    inputs, pair_id = prepare_probe_inputs(image_a, image_b)
    workspace.validate()
    infer, match, sources = import_probe_functions(backend.root,
        backend._fingerprint.values['source_fingerprint'])
    device = backend._device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    workspace.emit('probe_started', pair_id=pair_id, source='independent_pair_probe')
    try:
        # Probe no_grad is local; reconstruction still owns its enable_grad scope.
        with torch.no_grad():
            started = time.perf_counter()
            predictions = infer(backend._model, *inputs.images, device=backend._requested_device)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - started
            descs, confidences = validate_probe_predictions(predictions, image_a, image_b)
            if any(value.device != device for value in descs + confidences):
                raise BackendError('Observed probe prediction device differs from prepared model device')
            started = time.perf_counter()
            matches = match(descs, confidences, subsample=options.subsample,
                            device=backend._requested_device, ptmap_key='pred_desc')
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            matching_seconds = time.perf_counter() - started
            arrays = extract_probe_matches(matches, image_a, image_b, options)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == 'cuda' else None
        observation = PairObservation(pair_id, image_a.metadata(), image_b.metadata(), *arrays,
            option_values, backend._fingerprint, sources, inference_seconds, matching_seconds, peak,
            {'cache_namespace': workspace.cache_namespace, 'cache_mode': 'off',
             'cache_read': False, 'cache_write': False, 'scene_optimizer_called': False})
    except Exception as error:
        workspace.emit('probe_failed', pair_id=pair_id, error_type=type(error).__name__, error=str(error))
        raise
    workspace.emit('probe_completed', pair_id=pair_id,
                   raw_count=len(observation.xy_a), retained_count=int(observation.keep_mask.sum()))
    return observation
