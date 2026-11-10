"""Per-call capture of real sparse anchor pixels without a vendor/global patch."""
from types import FunctionType
import weakref

import numpy as np
import torch

from .sparse_io import BackendIOError, VendorInputs, SPARSE_SOURCE, SPARSE_SOURCE_SHA256


def _require(condition, message):
    if not condition:
        raise BackendIOError(message)


class SparsePixelCapture:
    """Keep original alignment code/helpers; replace only its constructor lookup.

The private function globals are per invocation. Helpers still reference their
original globals, and the vendor module is never mutated. The pinned source
contract establishes that make_pts3d and SparseGA use the same anchor row order.
We capture constructor arguments, never infer pixels from geometry or colors.
Only CPU pixel copies and a weak scene reference survive construction.
"""
    def __init__(self, alignment, inputs):
        _require(isinstance(inputs, VendorInputs), 'Expected prepared VendorInputs')
        _require(isinstance(alignment, FunctionType), 'Pinned alignment must be a Python function')
        constructor = alignment.__globals__.get('SparseGA')
        _require(isinstance(constructor, type) and constructor.__name__ == 'SparseGA'
                 and 'SparseGA' in alignment.__code__.co_names,
                 'Pinned alignment constructor lookup is missing')
        self._constructor = constructor
        self._inputs = inputs
        self._started = False
        self._pixels = None
        self._scene = None
        globals_copy = dict(alignment.__globals__)
        globals_copy['SparseGA'] = self._construct
        self.alignment = FunctionType(alignment.__code__, globals_copy,
            alignment.__name__, alignment.__defaults__, alignment.__closure__)
        self.alignment.__kwdefaults__ = (None if alignment.__kwdefaults__ is None
                                        else dict(alignment.__kwdefaults__))

    def _construct(self, img_paths, pairs_in, res_fine, anchors, canonical_paths=None):
        _require(not self._started, 'Sparse scene must be constructed exactly once per invocation')
        self._started = True
        inputs = self._inputs
        _require(isinstance(img_paths, (tuple, list)) and tuple(img_paths) == inputs.paths,
                 'Captured image paths/order differ from prepared inputs')
        count = len(inputs.paths)
        _require(isinstance(anchors, dict) and list(anchors) == list(range(count))
                 and all(type(key) is int for key in anchors),
                 'Anchor keys/insertion order differ from image order')
        _require(isinstance(res_fine, dict) and isinstance(res_fine.get('pts3d'), (tuple, list))
                 and len(res_fine['pts3d']) == count, 'Missing per-image optimized points')
        pixels = []
        for i, image in enumerate(inputs.images):
            entry = anchors[i]
            _require(isinstance(entry, (tuple, list)) and len(entry) == 3, 'Malformed sparse anchors')
            raw = entry[0]
            _require(isinstance(raw, torch.Tensor) and raw.dtype in (torch.int32, torch.int64)
                     and raw.ndim == 2 and raw.shape[1] == 3,
                     'Pinned sparse anchor coordinates must be integer M,3 homogeneous pixels')
            xyh = raw.detach().cpu().numpy()
            _require(np.all(xyh[:, 2] == 1), 'Invalid homogeneous anchor coordinate')
            xy = xyh[:, :2].astype(np.int64, copy=True)
            height, width = image['true_shape'][0]
            _require(np.all((xy >= 0) & (xy < [width, height])),
                     'Sparse anchor pixels exceed processed image bounds')
            points = res_fine['pts3d'][i]
            _require(getattr(points, 'shape', None) == (len(xy), 3),
                     'Sparse anchor rows differ from optimized point rows')
            pixels.append(xy)
        # Original constructor receives the unchanged arguments and shared objects.
        scene = self._constructor(img_paths, pairs_in, res_fine, anchors, canonical_paths)
        colors = scene.get_pts3d_colors()
        _require(isinstance(colors, (tuple, list)) and len(colors) == count,
                 'Captured scene colors lack one entry per image')
        for i, xy in enumerate(pixels):
            rgb_image = (inputs.images[i]['img'][0].permute(1, 2, 0).numpy() * .5 + .5).clip(0, 1)
            expected = rgb_image[xy[:, 1], xy[:, 0]]
            _require(isinstance(colors[i], np.ndarray) and np.array_equal(colors[i], expected),
                     'Sparse scene colors differ from captured source pixel rows')
        self._pixels = tuple(pixels)
        self._scene = weakref.ref(scene)
        return scene

    def pixels_for(self, scene):
        _require(self._pixels is not None and self._scene is not None and self._scene() is scene,
                 'Returned scene is not the one with captured sparse anchor pixels')
        return self._pixels

    def metadata(self):
        _require(self._pixels is not None, 'Sparse anchor capture did not complete')
        return {'basis': 'pinned_SparseGA_constructor_anchor_pixels',
            'source_file': SPARSE_SOURCE, 'source_sha256': SPARSE_SOURCE_SHA256,
            'coordinate_system': 'processed_image', 'columns': ['x', 'y'],
            'row_order': 'exact_per_image_anchor_rows_used_by_make_pts3d_and_SparseGA_colors',
            'duplicate_xy_preserved': True, 'owned_cpu_copy': True,
            'private_cache_read': False, 'vendor_module_globals_modified': False}
