"""Owned CPU reconstruction results, independent of model/vendor imports.

Captured anchor pixels may be supplied per view; every observation field shares
one filtering mask. Validated reconstruction confidence remains unavailable.
Missing data stays explicitly unavailable.
"""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np


class ResultError(ValueError):
    """Invalid geometry contract or empty retained point cloud."""


@dataclass(frozen=True)
class ReconstructionResult:
    image_ids: tuple
    points_world: np.ndarray
    colors_rgb: np.ndarray
    K_processed: np.ndarray
    T_world_from_camera: np.ndarray
    T_standard_world_from_upstream_world: np.ndarray
    source_image_index: np.ndarray
    source_observation_index: np.ndarray
    source_pixels_xy: object
    point_confidence: object
    processed_hw: object
    view_masks: tuple
    filtering: dict

    @property
    def reference_image_id(self):
        return self.image_ids[0]

    def metadata(self):
        def optional(value, reason):
            return ({'status': 'available'} if value is not None else
                    {'status': 'unavailable', 'reason': reason})
        return {
            'schema_version': 1, 'result_kind': 'reconstruction_result',
            'image_ids': list(self.image_ids), 'point_count': len(self.points_world),
            'camera_count': len(self.image_ids),
            'reference_image_id': self.reference_image_id, 'scale_status': 'unverified',
            'coordinate_convention': {
                'handedness': 'right', 'camera_axes': {'x': 'right', 'y': 'down', 'z': 'forward'},
                'vectors': 'column', 'pose': 'camera_to_standard_world',
                'world_reference': 'first_registered_camera_in_manifest_order',
                'intrinsics_pixels': 'processed_image', 'dimensions': 'H,W', 'pixels': 'x,y',
            },
            'T_standard_world_from_upstream_world':
                self.T_standard_world_from_upstream_world.tolist(),
            'processed_hw': None if self.processed_hw is None else [list(hw) for hw in self.processed_hw],
            'source_observation_index_meaning': 'zero-based row in each original sparse view; not a pixel coordinate',
            'availability': {
                'source_pixels_xy': optional(self.source_pixels_xy,
                    'Original sparse anchor coordinates were not supplied; no pixel coordinates are inferred'),
                'point_confidence': optional(self.point_confidence,
                    'No validated per-observation reconstruction confidence was supplied'),
            },
            'filtering': deepcopy(self.filtering), 'geometry_quality_accepted': False,
        }


def _require(condition, message):
    if not condition:
        raise ResultError(message)


def _float_array(value, shape, label, *, integer_allowed=False):
    kinds = 'fiu' if integer_allowed else 'f'
    _require(isinstance(value, np.ndarray) and value.dtype.kind in kinds,
             f'{label} must be a floating CPU numpy array')
    _require(value.shape == shape, f'{label} shape must be {shape}')
    return value.astype(np.float64, copy=True)


def normalize_sparse_scene(scene, *, pixels_xy=None, processed_hw=None,
                           confidence=None, confidence_field=None, confidence_threshold=None):
    """Apply one rigid world transform and one per-view observation mask.

Camera matrices are mandatory, finite, and rigid; malformed cameras are errors.
Nonfinite observation rows are removed jointly across all supplied row fields.
Available colors remain normalized RGB; invalid finite ranges are errors.
"""
    ids = getattr(scene, 'image_ids', None)
    views = getattr(scene, 'views', None)
    _require(isinstance(ids, (tuple, list)) and len(ids) > 0
             and all(isinstance(value, str) and value for value in ids), 'Invalid image IDs')
    ids = tuple(ids)
    _require(len(set(ids)) == len(ids), 'Duplicate image IDs')
    count = len(ids)
    _require(isinstance(views, (tuple, list)) and len(views) == count
             and tuple(getattr(view, 'image_id', None) for view in views) == ids,
             'View identities/count/order differ from image IDs')
    K = _float_array(getattr(scene, 'K_processed', None), (count, 3, 3), 'Intrinsics')
    poses = _float_array(getattr(scene, 'T_world_from_camera', None), (count, 4, 4), 'Camera poses')
    _require(np.isfinite(K).all() and np.isfinite(poses).all(), 'Nonfinite camera matrices')
    _require((K[:, (0, 1), (0, 1)] > 0).all()
             and np.allclose(K[:, 2], [0, 0, 1], rtol=0, atol=1e-6), 'Invalid camera intrinsics')
    _require(np.allclose(poses[:, 3], [0, 0, 0, 1], rtol=0, atol=1e-6), 'Invalid pose homogeneous row')
    rotations = poses[:, :3, :3]
    _require(np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), rtol=0, atol=1e-4)
             and np.allclose(np.linalg.det(rotations), 1., rtol=0, atol=1e-4),
             'Camera rotations must be proper rigid rotations')
    # Use the exact inverse of the accepted finite-precision reference pose.
    # This preserves relative camera geometry without snapping an approximate R.
    transform = np.linalg.inv(poses[0])
    with np.errstate(over='ignore', invalid='ignore'):
        normalized_poses = transform[None] @ poses
    _require(np.isfinite(transform).all() and np.isfinite(normalized_poses).all(),
             'World normalization produced nonfinite camera poses')

    if processed_hw is not None:
        _require(isinstance(processed_hw, (tuple, list)) and len(processed_hw) == count,
                 'Expected one processed H,W per image')
        _require(all(isinstance(hw, (tuple, list)) and len(hw) == 2
                     and all(type(size) is int and size > 0 for size in hw) for hw in processed_hw),
                 'Processed dimensions must be positive H,W integers')
        processed_hw = tuple(tuple(hw) for hw in processed_hw)
    for value, label in ((pixels_xy, 'Pixels'), (confidence, 'Confidence')):
        if value is not None:
            _require(isinstance(value, (tuple, list)) and len(value) == count,
                     f'Expected one {label} array per view')
    _require(pixels_xy is None or processed_hw is not None, 'Pixel bounds need processed H,W')
    if confidence is not None:
        _require(isinstance(confidence_field, str) and bool(confidence_field.strip()),
                 'Available confidence requires an explicit field name')
    else:
        _require(confidence_field is None and confidence_threshold is None,
                 'Confidence field/threshold cannot be used without confidence')
    if confidence_threshold is not None:
        _require(type(confidence_threshold) in (int, float)
                 and np.isfinite(confidence_threshold) and confidence_threshold >= 0,
                 'Confidence threshold must be finite and nonnegative')

    rows = {key: [] for key in ('points', 'colors', 'image_index', 'row_index', 'pixels', 'confidence')}
    masks, per_view = [], []
    input_count = removed_nonfinite = removed_confidence = 0
    for index, view in enumerate(views):
        raw = getattr(view, 'points_world', None)
        _require(isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.shape[1] == 3,
                 'Points must have shape M,3')
        n = len(raw)
        points = _float_array(raw, (n, 3), 'Points')
        colors = _float_array(getattr(view, 'colors_rgb', None), (n, 3), 'Colors')
        _require(not np.any(np.isfinite(colors) & ((colors < 0) | (colors > 1))),
                 'Finite RGB values must lie in [0,1]; no clipping')
        finite = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
        pixel = conf = None
        if pixels_xy is not None:
            pixel = _float_array(pixels_xy[index], (n, 2), 'Source pixels x,y', integer_allowed=True)
            valid_pixel = np.isfinite(pixel).all(axis=1)
            height, width = processed_hw[index]
            _require(((pixel[valid_pixel] >= 0) & (pixel[valid_pixel] < [width, height])).all(),
                     'Source pixels exceed processed image bounds')
            finite &= valid_pixel
        if confidence is not None:
            conf = _float_array(confidence[index], (n,), 'Point confidence')
            finite &= np.isfinite(conf)
        mask = finite.copy()
        if confidence_threshold is not None:
            mask &= conf >= confidence_threshold
        input_count += n
        removed_nonfinite += int((~finite).sum())
        removed_confidence += int((finite & ~mask).sum())
        with np.errstate(over='ignore', invalid='ignore'):
            world = points[mask] @ transform[:3, :3].T + transform[:3, 3]
        _require(np.isfinite(world).all(), 'World normalization produced nonfinite points')
        rows['points'].append(world)
        rows['colors'].append(colors[mask])
        rows['image_index'].append(np.full(int(mask.sum()), index, dtype=np.int64))
        rows['row_index'].append(np.arange(n, dtype=np.int64)[mask])
        if pixel is not None: rows['pixels'].append(pixel[mask])
        if conf is not None: rows['confidence'].append(conf[mask])
        masks.append(mask)
        per_view.append({'image_id': ids[index], 'input_count': n, 'retained_count': int(mask.sum())})
    retained = sum(item['retained_count'] for item in per_view)
    _require(retained > 0, 'Empty retained point cloud is a quality failure')
    filtering = {
        'input_count': input_count, 'retained_count': retained,
        'removed_nonfinite': removed_nonfinite, 'removed_confidence': removed_confidence,
        'retained_fraction': retained / input_count, 'per_view': per_view,
        'confidence_field': confidence_field, 'confidence_threshold': confidence_threshold,
        'confidence_strategy': 'disabled' if confidence_threshold is None else 'named_field_threshold',
        'threshold_comparison': '>=', 'same_mask_for_all_observation_fields': True,
    }
    return ReconstructionResult(ids, np.concatenate(rows['points']), np.concatenate(rows['colors']),
        K, normalized_poses, transform, np.concatenate(rows['image_index']),
        np.concatenate(rows['row_index']),
        None if pixels_xy is None else np.concatenate(rows['pixels']),
        None if confidence is None else np.concatenate(rows['confidence']),
        processed_hw, tuple(masks), filtering)
