"""Input/output boundary for the pinned MASt3R sparse-GA route.

This module never imports MASt3R, loads weights, runs optimization, or reads
private upstream caches. Runtime lifecycle lives in the separate mast3r adapter.
"""
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from ..pairing import to_directed_pairs, validate_pair_graph

SPARSE_SOURCE = 'vendor/mast3r/mast3r/cloud_opt/sparse_ga.py'
SPARSE_SOURCE_SHA256 = '195cc7510b65871f40ea51cb36df40313187fa3ac47c412576c755c44573ad74'


class BackendIOError(ValueError):
    """Invalid backend inputs or mandatory returned scene data."""


@dataclass(frozen=True)
class VendorInputs:
    """Fresh mutable vendor dictionaries; tensors are borrowed without copying."""
    image_ids: tuple
    paths: tuple
    images: tuple
    pairs: tuple
    pair_graph_hash: str

    def metadata(self):
        return {'image_ids': list(self.image_ids), 'pair_graph_hash': self.pair_graph_hash,
            'directed_item_count': len(self.pairs), 'instance_identity': 'resolved_file_path',
            'image_mapping': [{'image_id': image_id, 'idx': index, 'path': path}
                for index, (image_id, path) in enumerate(zip(self.image_ids, self.paths))],
            'pixel_copy': False, 'vendor_dictionaries_are_fresh': True}


def prepare_vendor_inputs(batch, pair_graph):
    """Keep manifest order and make dictionary mutation safe for own records.

Only dictionaries and true_shape arrays are newly allocated. Pixel tensors are
borrowed read-only for the pinned forward route; this is not tensor isolation.
"""
    validate_pair_graph(pair_graph)
    records = getattr(batch, 'records', None)
    if not isinstance(records, (list, tuple)):
        raise BackendIOError('Expected ImageBatch.records')
    image_ids = tuple(getattr(record, 'image_id', None) for record in records)
    if image_ids != pair_graph.nodes:
        raise BackendIOError('Image identities/order differ from PairGraph')
    paths, images = [], []
    for index, record in enumerate(records):
        if type(record.manifest_index) is not int or record.manifest_index != index:
            raise BackendIOError('Image manifest_index differs from record order')
        if not isinstance(record.path, Path) or not record.path.is_absolute():
            raise BackendIOError('ImageRecord.path must be an absolute Path')
        path = str(record.path.resolve())
        if path in paths:
            raise BackendIOError('Duplicate resolved image path')
        hw, tensor = record.processed_hw, record.model_tensor
        if (not isinstance(hw, tuple) or len(hw) != 2
                or any(type(value) is not int or value <= 0 for value in hw)):
            raise BackendIOError('Processed image dimensions must be positive H,W integers')
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.device.type != 'cpu' or tuple(tensor.shape) != (1, 3, *hw)
                or not bool(torch.isfinite(tensor).all())
                or not bool(((tensor >= -1) & (tensor <= 1)).all())):
            raise BackendIOError(f'Invalid CPU FP32 normalized image tensor: {record.image_id}')
        paths.append(path)
        images.append(dict(img=tensor, true_shape=np.array([hw], dtype=np.int32),
                           idx=index, instance=path))
    pairs = to_directed_pairs(pair_graph, images, image_ids=image_ids)
    return VendorInputs(image_ids, tuple(paths), tuple(images), tuple(pairs), pair_graph.graph_hash)


@dataclass(frozen=True)
class SparseViewData:
    image_id: str
    points_world: np.ndarray
    colors_rgb: np.ndarray
    pixels_xy: object = None


@dataclass(frozen=True)
class SparseSceneData:
    """Owned CPU copies detached from the vendor scene and its cache lifetime.

Arrays remain mutable. Observation counts include repeated image observations
of the same surface and are not distinct physical points or quality scores.
"""
    image_ids: tuple
    views: tuple
    K_processed: np.ndarray
    T_world_from_camera: np.ndarray
    pixel_source: object = None

    @property
    def pixels_xy(self):
        values = tuple(view.pixels_xy for view in self.views)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise BackendIOError('Only some sparse views have source pixels')
        return values

    @property
    def points_world(self):
        return np.concatenate([view.points_world for view in self.views])

    @property
    def colors_rgb(self):
        return np.concatenate([view.colors_rgb for view in self.views])

    @property
    def observation_count(self):
        return sum(len(view.points_world) for view in self.views)

    def metadata(self):
        reasons = {
            'dense_depth': 'Anchor optimization depths are not complete processed-image depth maps',
            'point_confidence': 'No validated per-observation confidence extraction in this step',
            'pair_matching_metrics': 'No validated full-scene pair-metric extraction in this step',
            'actual_model_forward_calls': 'Directed item count does not observe inference/cache execution',
        }
        return {'schema_version': 1, 'result_kind': 'sparse_scene_data',
            'image_ids': list(self.image_ids), 'observation_count': self.observation_count,
            'per_image_observation_counts': [len(view.points_world) for view in self.views],
            'K_processed_shape': list(self.K_processed.shape),
            'T_world_from_camera_shape': list(self.T_world_from_camera.shape),
            'coordinate_convention': {'intrinsics': 'processed_image_pixel_coordinates',
                'pose': 'camera_to_upstream_world', 'points': 'upstream_world',
                'world_gauge': 'upstream_optimizer; no additional canonicalization',
                'scale': 'unresolved', 'metric_units_verified': False},
            'source_pixels_xy': ({'status': 'available',
                'source': deepcopy(self.pixel_source) if self.pixel_source is not None
                          else {'basis': 'supplied_per_view_arrays; extraction_not_verified'}}
                if self.pixels_xy is not None else {'status': 'unavailable',
                    'reason': 'Original sparse anchor pixels were not supplied; no inference'}),
            'unavailable': {name: {'status': 'unavailable', 'value': None, 'reason': reason}
                            for name, reason in reasons.items()},
            'geometry_quality_accepted': False}


def _owned_array(value, label):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if not isinstance(value, np.ndarray) or value.dtype.kind != 'f':
        raise BackendIOError(f'{label} must be a floating tensor or array')
    result = value.copy()
    if not np.isfinite(result).all():
        raise BackendIOError(f'{label} contains nonfinite values')
    return result


def extract_sparse_scene(scene, inputs, *, pixels_xy=None, pixel_source=None):
    """Extract required fields before the upstream temporary workspace closes.

The pinned make_pts3d applies cam2w to each image's observations. Depth and
confidence are deliberately not extracted through private cache formats here.
"""
    if not isinstance(inputs, VendorInputs):
        raise BackendIOError('Expected prepared VendorInputs')
    try:
        returned_paths = scene.img_paths
        if not isinstance(returned_paths, (list, tuple)) or tuple(returned_paths) != inputs.paths:
            raise BackendIOError('Returned scene image paths/order differ from prepared inputs')
        points = scene.get_sparse_pts3d()
        colors = scene.get_pts3d_colors()
        poses = _owned_array(scene.get_im_poses(), 'Camera poses')
        intrinsics = _owned_array(scene.intrinsics, 'Camera intrinsics')
    except AttributeError as error:
        raise BackendIOError(f'Mandatory sparse scene field is missing: {error}') from error
    count = len(inputs.image_ids)
    if pixels_xy is not None and (not isinstance(pixels_xy, (tuple, list)) or len(pixels_xy) != count):
        raise BackendIOError('Expected one captured pixel array per image')
    if pixels_xy is None and pixel_source is not None:
        raise BackendIOError('Pixel provenance cannot exist without source pixels')
    if (not isinstance(points, (list, tuple)) or not isinstance(colors, (list, tuple))
            or len(points) != count or len(colors) != count):
        raise BackendIOError('Sparse points/colors must contain one entry per image')
    if poses.shape != (count, 4, 4):
        raise BackendIOError('Camera poses must have shape N,4,4')
    if (intrinsics.shape != (count, 3, 3)
            or not (intrinsics[:, (0, 1), (0, 1)] > 0).all()):
        raise BackendIOError('Camera intrinsics must have shape N,3,3 with positive focals')
    views = []
    for index, (image_id, pts, rgb) in enumerate(zip(inputs.image_ids, points, colors)):
        pts = _owned_array(pts, f'Points for {image_id}')
        rgb = _owned_array(rgb, f'Colors for {image_id}')
        if pts.ndim != 2 or pts.shape[1] != 3 or rgb.shape != pts.shape:
            raise BackendIOError(f'Per-image points/colors must share shape M,3: {image_id}')
        xy = None
        if pixels_xy is not None:
            value = pixels_xy[index]
            if (not isinstance(value, np.ndarray) or value.dtype.kind not in 'iu'
                    or value.shape != (len(pts), 2)):
                raise BackendIOError('Captured pixel arrays must be integer M,2 matching point rows')
            height, width = inputs.images[index]['true_shape'][0]
            if not np.all((value >= 0) & (value < [width, height])):
                raise BackendIOError('Captured pixels exceed processed image bounds')
            xy = value.copy()
        views.append(SparseViewData(image_id, pts, rgb, xy))
    if sum(len(view.points_world) for view in views) == 0:
        raise BackendIOError('Sparse scene contains no observations')
    return SparseSceneData(inputs.image_ids, tuple(views), intrinsics, poses, deepcopy(pixel_source))
