"""Own image records over the pinned DUSt3R 512/patch-16 policy.

Pixel processing delegates to vendor/mast3r/dust3r/dust3r/utils/image.py.
The rounding/crop metadata follows that file, Copyright Naver Corporation,
CC BY-NC-SA 4.0; its source and SHA256 are retained in each batch's evidence.
This module does not modify or redistribute a new vendor snapshot.
"""
from dataclasses import dataclass
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image
import torch

POLICY = "dust3r_compatible_512_patch16_v1"
ROOT = Path(__file__).resolve().parents[2]
_EXIF_OPERATIONS = {1: "none", 2: "mirror_horizontal", 3: "rotate_180",
    4: "mirror_vertical", 5: "transpose", 6: "rotate_90_clockwise",
    7: "transverse", 8: "rotate_90_counterclockwise"}


class ImageLoadError(ValueError):
    """Invalid images or an inconsistency with the pinned preprocessing policy."""


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _array_hash(array):
    header = json.dumps({"shape": list(array.shape), "dtype": str(array.dtype)},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\0" + np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_manifest(path):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ImageLoadError(f"Duplicate manifest key: {key}")
            result[key] = value
        return result
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    except (OSError, UnicodeError, ValueError) as error:
        raise ImageLoadError(f"Cannot read scene manifest {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("images"), list) or not value["images"]:
        raise ImageLoadError("Scene manifest must contain a nonempty images list")
    return value


def _upstream_loader():
    package_root = ROOT / "vendor/mast3r/dust3r"
    source = package_root / "dust3r/utils/image.py"
    sys.path.insert(0, str(package_root))
    module = importlib.import_module("dust3r.utils.image")
    if Path(module.__file__).resolve() != source.resolve():
        raise ImageLoadError("Image loader import is outside the pinned vendor file")
    return module.load_images, {"file": source.relative_to(ROOT).as_posix(),
                                "sha256": _file_hash(source)}


def _geometry(path, square_ok):
    # Inspect headers/EXIF only; do not resize/crop or normalize a second image.
    try:
        with Image.open(path) as image:
            if image.format not in ("JPEG", "PNG"):
                raise ImageLoadError(f"Only JPG/PNG decoding is supported: {path.name}")
            width, height = image.size
            orientation = image.getexif().get(274, 1)
    except (OSError, ValueError) as error:
        raise ImageLoadError(f"Cannot decode image header {path.name}: {error}") from error
    if type(orientation) is not int or orientation not in _EXIF_OPERATIONS:
        raise ImageLoadError(f"Invalid EXIF orientation: {path.name}")
    original_hw = (height, width)
    if orientation >= 5:
        width, height = height, width
    oriented_hw = (height, width)
    long_side = max(width, height)
    # Keep the same floating-point operation order as the pinned loader.
    resized_width = int(round(width * 512 / long_side))
    resized_height = int(round(height * 512 / long_side))
    if min(resized_width, resized_height) < 16:
        raise ImageLoadError(f"Aspect ratio leaves no patch-16 crop: {path.name}")
    center_x, center_y = resized_width // 2, resized_height // 2
    half_width = ((2 * center_x) // 16) * 8
    half_height = ((2 * center_y) // 16) * 8
    if not square_ok and resized_width == resized_height:
        half_height = 3 * half_width // 4
    crop = (center_x - half_width, center_y - half_height,
            center_x + half_width, center_y + half_height)
    processed_hw = (2 * half_height, 2 * half_width)
    sx, sy = resized_width / width, resized_height / height
    transform = np.array([[sx, 0, (sx - 1) / 2 - crop[0]],
                          [0, sy, (sy - 1) / 2 - crop[1]], [0, 0, 1]], dtype=np.float64)
    return original_hw, oriented_hw, orientation, (resized_height, resized_width), crop, processed_hw, transform


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    relative_path: str
    path: Path
    file_sha256: str
    manifest_index: int
    original_hw: tuple
    oriented_hw: tuple
    exif_orientation: int
    resize_hw: tuple
    crop_xyxy: tuple
    processed_hw: tuple
    A_oriented_to_processed: np.ndarray
    processed_rgb: np.ndarray
    model_tensor: torch.Tensor
    processed_pixel_sha256: str
    model_tensor_sha256: str
    schema_version: int = 1

    def metadata(self):
        return {"schema_version": self.schema_version, "image_id": self.image_id,
            "relative_path": self.relative_path, "file_sha256": self.file_sha256,
            "manifest_index": self.manifest_index, "original_hw": list(self.original_hw),
            "oriented_hw": list(self.oriented_hw), "exif_orientation": self.exif_orientation,
            "exif_operation": _EXIF_OPERATIONS[self.exif_orientation],
            "resize_hw": list(self.resize_hw), "crop_xyxy": list(self.crop_xyxy),
            "processed_hw": list(self.processed_hw), "preprocess_policy": POLICY,
            "A_oriented_to_processed": self.A_oriented_to_processed.tolist(),
            "processed_pixel_sha256": self.processed_pixel_sha256,
            "model_tensor_sha256": self.model_tensor_sha256}


@dataclass(frozen=True)
class ImageBatch:
    records: list
    manifest_sha256: str
    processed_pixel_hash: str
    warnings: list
    upstream_source: dict
    preprocess_config: dict
    schema_version: int = 1
    pairing_declaration: object = None

    def write_evidence(self, directory, *, save_processed=False):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": self.schema_version, "preprocess_policy": POLICY,
            "preprocess_config": self.preprocess_config, "manifest_sha256": self.manifest_sha256,
            "processed_pixel_hash": self.processed_pixel_hash, "warnings": self.warnings,
            "upstream_source": self.upstream_source,
            "coordinate_convention": {"dimensions": "H,W", "pixels": "x,y",
                "resize": "half_pixel", "crop_box": "left,top,right,bottom; right/bottom exclusive",
                "transform_input": "EXIF-oriented integer pixel centers"},
            "records": [record.metadata() for record in self.records]}
        if self.pairing_declaration is not None:
            from copy import deepcopy
            data['pairing_declaration'] = deepcopy(self.pairing_declaration)
        if save_processed:
            processed = directory / "processed"
            processed.mkdir(parents=True, exist_ok=True)
            for index, record in enumerate(self.records):
                name = f"processed/{index:04d}.png"
                Image.fromarray(record.processed_rgb).save(directory / name)
                data["records"][index]["debug_processed_png"] = name
        (directory / "image_preprocessing.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load_scene(scene_manifest, preprocess_config):
    if (not isinstance(preprocess_config, dict)
            or set(preprocess_config) != {"long_edge", "square_ok"}
            or type(preprocess_config["long_edge"]) is not int or preprocess_config["long_edge"] != 512
            or type(preprocess_config["square_ok"]) is not bool):
        raise ImageLoadError("Current preprocessing accepts only long_edge=512 and boolean square_ok")
    manifest_path = Path(scene_manifest).resolve()
    manifest = _read_manifest(manifest_path)
    pending, seen_ids, seen_paths, hashes, warnings = [], set(), set(), {}, []
    for index, item in enumerate(manifest["images"]):
        if not isinstance(item, dict):
            raise ImageLoadError("Each image entry must be a mapping")
        relative_path = item.get("file")
        if (not isinstance(relative_path, str) or not relative_path.strip()
                or Path(relative_path).is_absolute()):
            raise ImageLoadError("Image file must be a nonempty relative path")
        path = (manifest_path.parent / relative_path).resolve()
        if not path.is_relative_to(manifest_path.parent) or not path.is_file():
            raise ImageLoadError(f"Missing image or path outside manifest directory: {relative_path}")
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            raise ImageLoadError(f"Unsupported image extension: {relative_path}")
        image_id = item.get("image_id", relative_path)
        if not isinstance(image_id, str) or not image_id.strip():
            raise ImageLoadError("Image ID must be a nonempty string")
        if image_id in seen_ids or path in seen_paths:
            raise ImageLoadError(f"Duplicate image ID or file path: {relative_path}")
        seen_ids.add(image_id)
        seen_paths.add(path)
        expected = item.get("sha256")
        if not isinstance(expected, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None:
            raise ImageLoadError(f"Invalid image SHA256: {relative_path}")
        digest = _file_hash(path)
        if digest != expected.lower():
            raise ImageLoadError(f"Image SHA256 mismatch: {relative_path}")
        if digest in hashes:
            warnings.append(f"Identical image contents: {hashes[digest]} and {image_id}")
        hashes[digest] = image_id
        geometry = _geometry(path, preprocess_config["square_ok"])
        pending.append((image_id, relative_path, path, digest, index, geometry))
    from .pairing import validate_pairing_declaration, PairingError
    declaration = None
    if 'pairing' in manifest:
        try:
            declaration = validate_pairing_declaration(manifest['pairing'], tuple(item[0] for item in pending))
        except PairingError as error:
            raise ImageLoadError(str(error)) from error
    loader, upstream_source = _upstream_loader()
    try:
        upstream_images = loader([str(item[2]) for item in pending], size=512,
            square_ok=preprocess_config["square_ok"], verbose=False, patch_size=16)
    except Exception as error:
        raise ImageLoadError(f"Pinned image loader failed: {error}") from error
    if len(upstream_images) != len(pending):
        raise ImageLoadError("Pinned image loader skipped or added an image")
    records = []
    for item, upstream in zip(pending, upstream_images):
        image_id, relative_path, path, digest, index, geometry = item
        original_hw, oriented_hw, orientation, resize_hw, crop, processed_hw, transform = geometry
        tensor = upstream["img"]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32
                or tensor.device.type != "cpu" or tuple(tensor.shape) != (1, 3, *processed_hw)
                or not bool(torch.isfinite(tensor).all())
                or not bool(((tensor >= -1) & (tensor <= 1)).all())
                or not np.array_equal(upstream["true_shape"], np.array([processed_hw], dtype=np.int32))
                or upstream["idx"] != index):
            raise ImageLoadError(f"Pinned tensor/shape/order differs from metadata: {relative_path}")
        rgb = np.rint(np.clip(tensor[0].detach().numpy().transpose(1, 2, 0) * 0.5 + 0.5, 0, 1)
                      * 255).astype(np.uint8)
        records.append(ImageRecord(image_id, relative_path, path, digest, index, original_hw,
            oriented_hw, orientation, resize_hw, crop, processed_hw, transform, rgb, tensor,
            _array_hash(rgb), _array_hash(tensor.detach().numpy())))
    identity = {"policy": POLICY, "preprocess_config": preprocess_config,
        "records": [{"image_id": record.image_id,
                     "processed_pixel_sha256": record.processed_pixel_sha256} for record in records]}
    batch_hash = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":")).encode("utf-8")).hexdigest()
    return ImageBatch(records, _file_hash(manifest_path), batch_hash, warnings, upstream_source,
                      dict(preprocess_config), pairing_declaration=declaration)
