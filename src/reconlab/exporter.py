"""CPU-only artifact export with verified rows and atomic directory publication."""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .results import ReconstructionResult


class ExportError(ValueError):
    """Invalid standard result, failed readback, or refused publication."""


@dataclass(frozen=True)
class ArtifactIndex:
    values: dict
    index_sha256: str

    def metadata(self):
        return {**deepcopy(self.values), 'index_sha256': self.index_sha256}


def _require(condition, message):
    if not condition: raise ExportError(message)


def _array(value, shape, kinds, label):
    _require(isinstance(value, np.ndarray) and value.shape == shape
             and value.dtype.kind in kinds and np.isfinite(value).all(),
             f'Invalid finite {label} array; expected shape {shape}')


def _validate(result):
    ids = result.image_ids
    _require(isinstance(ids, tuple) and len(ids) > 0
             and all(isinstance(value, str) and value for value in ids)
             and len(set(ids)) == len(ids), 'Invalid unique ordered image IDs')
    count = len(ids)
    _require(isinstance(result.points_world, np.ndarray) and result.points_world.ndim == 2,
             'Invalid points array')
    n = len(result.points_world)
    _require(n > 0, 'Empty point cloud is a quality failure')
    _array(result.points_world, (n, 3), 'f', 'points')
    _array(result.colors_rgb, (n, 3), 'f', 'RGB')
    _require(((result.colors_rgb >= 0) & (result.colors_rgb <= 1)).all(), 'RGB range exceeds [0,1]')
    _array(result.K_processed, (count, 3, 3), 'f', 'intrinsics')
    _array(result.T_world_from_camera, (count, 4, 4), 'f', 'camera poses')
    _array(result.T_standard_world_from_upstream_world, (4, 4), 'f', 'world transform')
    world_transform = result.T_standard_world_from_upstream_world
    _require(np.allclose(world_transform[3], [0, 0, 0, 1], rtol=0, atol=1e-6)
             and np.allclose(world_transform[:3, :3].T @ world_transform[:3, :3],
                             np.eye(3), rtol=0, atol=1e-4)
             and np.isclose(np.linalg.det(world_transform[:3, :3]), 1., rtol=0, atol=1e-4),
             'Invalid rigid world transform')
    K, poses = result.K_processed, result.T_world_from_camera
    _require((K[:, (0, 1), (0, 1)] > 0).all()
             and np.allclose(K[:, 2], [0, 0, 1], rtol=0, atol=1e-6), 'Invalid processed intrinsics')
    rotations = poses[:, :3, :3]
    _require(np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-6)
             and np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0, atol=1e-4)
             and np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-4)
             and np.allclose(poses[0], np.eye(4), rtol=0, atol=1e-8), 'Invalid normalized rigid camera poses')
    _array(result.source_image_index, (n,), 'iu', 'source image indices')
    _array(result.source_observation_index, (n,), 'iu', 'source row indices')
    _require(len(result.view_masks) == count and all(isinstance(mask, np.ndarray)
             and mask.ndim == 1 and mask.dtype == np.bool_ for mask in result.view_masks), 'Invalid per-view masks')
    expected_images = np.repeat(np.arange(count), [int(mask.sum()) for mask in result.view_masks])
    expected_rows = np.concatenate([np.flatnonzero(mask) for mask in result.view_masks])
    _require(np.array_equal(result.source_image_index, expected_images)
             and np.array_equal(result.source_observation_index, expected_rows),
             'Point source IDs/rows do not follow the original common masks')
    filtering = result.filtering
    _require(filtering['input_count'] == sum(len(mask) for mask in result.view_masks)
             and filtering['retained_count'] == n
             and filtering['removed_nonfinite'] >= 0 and filtering['removed_confidence'] >= 0
             and filtering['input_count'] - n == filtering['removed_nonfinite'] + filtering['removed_confidence']
             and np.isclose(filtering['retained_fraction'], n / filtering['input_count'], rtol=0, atol=1e-12)
             and filtering['same_mask_for_all_observation_fields'] is True,
             'Filtering counts/common-mask declaration differ')
    _require(len(filtering['per_view']) == count and all(item['image_id'] == ids[i]
             and item['input_count'] == len(result.view_masks[i])
             and item['retained_count'] == int(result.view_masks[i].sum())
             for i, item in enumerate(filtering['per_view'])), 'Per-view filtering metadata differs')
    if result.processed_hw is not None:
        _require(len(result.processed_hw) == count and all(len(hw) == 2
                 and all(type(size) is int and size > 0 for size in hw) for hw in result.processed_hw),
                 'Invalid processed H,W')
    if result.source_pixels_xy is not None:
        _array(result.source_pixels_xy, (n, 2), 'fiu', 'source pixels')
        _require(result.processed_hw is not None, 'Source pixels require processed bounds')
        hw = np.asarray(result.processed_hw)[result.source_image_index]
        _require(((result.source_pixels_xy >= 0) & (result.source_pixels_xy < hw[:, ::-1])).all(),
                 'Source pixel coordinates outside their images')
    if result.point_confidence is not None:
        _array(result.point_confidence, (n,), 'f', 'confidence')
        _require(isinstance(filtering['confidence_field'], str) and bool(filtering['confidence_field']),
                 'Confidence field must be named')
    else:
        _require(filtering['confidence_field'] is None and filtering['confidence_threshold'] is None,
                 'Unavailable confidence cannot declare a threshold')
    return np.rint(result.colors_rgb * 255).astype(np.uint8)


def _header(n):
    return ('ply\nformat binary_little_endian 1.0\ncomment reconlab standard world\n'
            f'element vertex {n}\nproperty double x\nproperty double y\nproperty double z\n'
            'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode('ascii')


def _ply_dtype():
    return np.dtype([(key, '<f8') for key in ('x', 'y', 'z')] +
                    [(key, 'u1') for key in ('red', 'green', 'blue')])


def _write_ply(path, points, rgb):
    rows = np.empty(len(points), dtype=_ply_dtype())
    for i, key in enumerate(('x', 'y', 'z')): rows[key] = points[:, i]
    for i, key in enumerate(('red', 'green', 'blue')): rows[key] = rgb[:, i]
    with path.open('xb') as stream:
        stream.write(_header(len(points)))
        stream.write(rows.tobytes())
        stream.flush()
        os.fsync(stream.fileno())


def read_scene_ply(path):
    """Read the exporter format, rejecting layout changes and partial bodies."""
    try:
        data = Path(path).read_bytes()
        end = data.find(b'end_header\n', 0, 65536)
        _require(end >= 0, 'Missing bounded PLY header')
        header = data[:end + len(b'end_header\n')]
        lines = header.decode('ascii').splitlines()
        _require(len(lines) > 3 and lines[3].startswith('element vertex '), 'Invalid PLY vertex declaration')
        n = int(lines[3].split()[-1])
        _require(n > 0 and header == _header(n), 'PLY header layout differs')
        body = data[len(header):]
        _require(len(body) == n * _ply_dtype().itemsize, 'PLY byte length/count differs')
        rows = np.frombuffer(body, dtype=_ply_dtype(), count=n)
        points = np.column_stack([rows[key] for key in ('x', 'y', 'z')])
        rgb = np.column_stack([rows[key] for key in ('red', 'green', 'blue')])
        _require(np.isfinite(points).all(), 'PLY contains nonfinite points')
        return points, rgb
    except (OSError, ValueError, UnicodeError) as error:
        if isinstance(error, ExportError): raise
        raise ExportError(f'PLY readback failed: {error}') from error


def _write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _write_cameras(path, value):
    _write_json(path, value)


def _publish(temporary, destination):
    _require(not destination.exists(), 'Artifact directory already exists')
    temporary.rename(destination)


def export_result(result, export_config, run_dir):
    """Publish run_dir/artifacts as one directory after every readback succeeds.

The pipeline owns run_dir creation. The exporter never overwrites an existing
artifact directory, infers missing fields, adds filtering, or changes geometry.
"""
    _require(isinstance(export_config, dict) and set(export_config) <= {'coordinate_dtype'}
             and export_config.get('coordinate_dtype', 'float64') == 'float64',
             'Only the verified float64 PLY coordinate export is supported')
    _require(isinstance(result, ReconstructionResult), 'Expected ReconstructionResult')
    result = deepcopy(result)
    try: rgb = _validate(result)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ExportError(f'Invalid reconstruction result: {error}') from error
    root = Path(run_dir).resolve()
    _require(root.is_dir(), 'Pipeline must create run_dir before export')
    destination = root / 'artifacts'
    _require(not destination.exists(), 'Artifact directory already exists')
    lock = root / '.export.lock'
    temporary = None
    owns_lock = False
    try:
        with lock.open('xb') as stream: stream.write(b'reconlab exporter\n')
        owns_lock = True
        temporary = Path(tempfile.mkdtemp(prefix='.export-', dir=root))
        metadata = result.metadata()
        camera_data = {
            'schema_version': 1, 'reference_image_id': result.reference_image_id,
            'scale_status': 'unverified', 'coordinate_convention': metadata['coordinate_convention'],
            'T_standard_world_from_upstream_world': metadata['T_standard_world_from_upstream_world'],
            'cameras': [{'image_id': image_id, 'manifest_index': i,
                'processed_hw': None if result.processed_hw is None else list(result.processed_hw[i]),
                'K_processed': result.K_processed[i].tolist(),
                'T_world_from_camera': result.T_world_from_camera[i].tolist(),
                'T_camera_from_world': np.linalg.inv(result.T_world_from_camera[i]).tolist()}
                for i, image_id in enumerate(result.image_ids)],
        }
        sources = {'image_ids': np.asarray(result.image_ids),
            'source_image_index': result.source_image_index,
            'source_observation_index': result.source_observation_index}
        sources.update({f'view_mask_{i:03d}': mask for i, mask in enumerate(result.view_masks)})
        if result.source_pixels_xy is not None: sources['source_pixels_xy'] = result.source_pixels_xy
        if result.point_confidence is not None: sources['point_confidence'] = result.point_confidence
        _write_ply(temporary / 'scene.ply', result.points_world, rgb)
        _write_cameras(temporary / 'cameras.json', camera_data)
        with (temporary / 'point_sources.npz').open('xb') as stream:
            np.savez_compressed(stream, **sources)
            stream.flush()
            os.fsync(stream.fileno())
        points_back, rgb_back = read_scene_ply(temporary / 'scene.ply')
        checks = {
            'point_count_and_coordinates': bool(np.array_equal(points_back, result.points_world)),
            'rgb_row_correspondence': bool(np.array_equal(rgb_back, rgb)),
            'coordinate_range': bool(np.array_equal(points_back.min(axis=0), result.points_world.min(axis=0))
                                     and np.array_equal(points_back.max(axis=0), result.points_world.max(axis=0))),
            'camera_ids_matrices_and_conventions':
                json.loads((temporary / 'cameras.json').read_bytes()) == camera_data,
        }
        with np.load(temporary / 'point_sources.npz', allow_pickle=False) as saved:
            checks['source_rows_masks_and_optional_fields'] = set(saved.files) == set(sources) and all(
                np.array_equal(saved[key], value) for key, value in sources.items())
        _require(all(checks.values()), 'Artifact readback failed; no directory published')
        artifacts = {}
        for name in ('scene.ply', 'cameras.json', 'point_sources.npz'):
            data = (temporary / name).read_bytes()
            artifacts[name] = {'path': f'artifacts/{name}', 'sha256': hashlib.sha256(data).hexdigest(),
                               'bytes': len(data)}
        values = {
            'schema_version': 1, 'kind': 'artifact_index', 'artifacts': artifacts,
            'index_file': 'artifacts/artifacts.json', 'point_count': len(result.points_world),
            'camera_count': len(result.image_ids), 'ply_coordinate_dtype': 'float64',
            'rgb_encoding': 'uint8; round-to-nearest-even from normalized RGB; maximum error 0.5/255',
            'result': metadata, 'readback_checks': checks,
            'publication': 'atomic rename of verified new artifact directory',
            'geometry_quality_accepted': False,
        }
        _write_json(temporary / 'artifacts.json', values)
        _require(json.loads((temporary / 'artifacts.json').read_bytes()) == values, 'Index readback differs')
        index_sha = hashlib.sha256((temporary / 'artifacts.json').read_bytes()).hexdigest()
        _publish(temporary, destination)
        temporary = None
        return ArtifactIndex(values, index_sha)
    except (OSError, ValueError, KeyError, TypeError) as error:
        if isinstance(error, ExportError): raise
        raise ExportError(f'Artifact export failed: {error}') from error
    finally:
        if temporary is not None and temporary.exists():
            # Verify the absolute target before recursively deleting our own staging directory.
            _require(temporary.resolve().parent == root and temporary.name.startswith('.export-'),
                     'Unsafe temporary cleanup target')
            shutil.rmtree(temporary)
        if owns_lock: lock.unlink()
