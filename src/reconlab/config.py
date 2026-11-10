"""Resolve the currently verified scene schema before model execution.

Inheritance is left to right, then the declaring file, then typed CLI overrides.
All current resource paths must remain within project_root. CLI paths are relative
to project_root; YAML paths are relative to the file that declares the value.
No environment substitution, YAML aliases, or implicit defaults are supported.
"""
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

import yaml


class ConfigError(ValueError):
    """Invalid or unsupported configuration; no silent fallback."""


class _StrictLoader(yaml.SafeLoader):
    def compose_node(self, parent, index):
        if self.check_event(yaml.AliasEvent):
            raise ConfigError("YAML aliases are not supported")
        return super().compose_node(parent, index)

    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("YAML keys must be strings")
            if key in result:
                raise ConfigError(f"Duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


# Types and supported values of the actual baseline, not future design options.
_FIELDS = {
    "schema_version": "version",
    "scene.id": "identifier", "scene.manifest": "path",
    "runtime.device": "device", "runtime.precision": ("fp32",),
    "preprocess.long_edge": (512,), "preprocess.square_ok": "bool",
    "pairing.strategy": ("complete", "sequential_window", "explicit"),
    "pairing.window": "positive_int",
    "backend.name": ("mast3r_sparse_ga",), "backend.weights_manifest": "path",
    "backend.shared_intrinsics": "bool", "backend.subsample": "positive_int",
    "backend.coarse_iterations": "positive_int",
    "backend.refine_iterations": "positive_int",
    "backend.coarse_lr": "positive_float", "backend.refine_lr": "positive_float",
    "backend.matching_conf_thr": "nonnegative_float", "output.root": "path",
}
_PATHS = {key for key, kind in _FIELDS.items() if kind == "path"}
_SECTIONS = {key.split(".")[0] for key in _FIELDS if "." in key}


def _validate(key, value):
    if key not in _FIELDS:
        raise ConfigError(f"Unknown configuration field: {key}")
    kind = _FIELDS[key]
    valid = False
    if isinstance(kind, tuple):
        valid = any(type(value) is type(item) and value == item for item in kind)
    elif kind == "version":
        valid = type(value) is int and value == 1
    elif kind == "bool":
        valid = type(value) is bool
    elif kind == "positive_int":
        valid = type(value) is int and value > 0
    elif kind in ("positive_float", "nonnegative_float"):
        valid = type(value) in (int, float) and math.isfinite(value)
        valid = valid and (value > 0 if kind == "positive_float" else value >= 0)
        if valid:
            value = float(value)
    elif kind == "identifier":
        valid = isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]+", value) is not None
    elif kind == "device":
        valid = isinstance(value, str) and re.fullmatch(r"cpu|cuda(?::[0-9]+)?", value) is not None
    elif kind == "path":
        valid = isinstance(value, str) and bool(value.strip()) and "$" not in value
    if not valid:
        raise ConfigError(f"Invalid or unsupported value for {key}: {value!r}")
    return value


def _fragment(document):
    if not isinstance(document, dict):
        raise ConfigError("Configuration must be a YAML mapping")
    if "schema_version" not in document:
        raise ConfigError("Every YAML file must declare schema_version")
    fields = {}
    for key, value in document.items():
        if key == "extends":
            continue
        if key in _SECTIONS:
            if not isinstance(value, dict):
                raise ConfigError(f"{key} must be a mapping")
            for child, item in value.items():
                name = f"{key}.{child}"
                fields[name] = _validate(name, item)
        else:
            fields[key] = _validate(key, value)
    return fields


def _relative_id(path, root):
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ConfigError(f"Current schema requires paths inside project_root: {path}") from error


def _nested(fields):
    result = {}
    for key, value in fields.items():
        if "." in key:
            section, child = key.split(".", 1)
            result.setdefault(section, {})[child] = value
        else:
            result[key] = value
    return result


def _check_device(device):
    if device == "cpu":
        return
    try:
        import torch
    except ImportError as error:
        raise ConfigError("CUDA was requested, but PyTorch cannot be imported") from error
    if not torch.cuda.is_available():
        raise ConfigError("CUDA was requested, but CUDA is unavailable; no CPU fallback")
    index = int(device.split(":", 1)[1]) if ":" in device else 0
    if index >= torch.cuda.device_count():
        raise ConfigError(f"CUDA device index is unavailable: {device}")


@dataclass(frozen=True)
class ResolvedConfig:
    values: dict
    sources: dict
    resource_ids: dict
    semantic_hash: str
    load_order: list

    def write_evidence(self, directory):
        """Call before reconstruction; both filenames are part of the contract."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "resolved_config.yaml").write_text(
            yaml.safe_dump(self.values, allow_unicode=True, sort_keys=False), encoding="utf-8")
        metadata = {"schema_version": 1, "semantic_hash": self.semantic_hash,
                    "resource_ids": self.resource_ids, "load_order": self.load_order,
                    "fields": self.sources}
        (directory / "config_sources.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8")


def resolve_config(config_path, cli_overrides=None, *, project_root):
    """CLI overrides are a mapping of dotted field names to already typed scalars.

    This hash identifies configuration semantics, not file contents or a cache key.
    Resource content hashes remain the responsibility of asset/run verification.
    """
    root = Path(project_root).resolve()
    values, sources, resource_ids, load_order = {}, {}, {}, []
    active = []

    def apply(fields, raw_fields, source, base):
        for key, value in fields.items():
            sources.setdefault(key, []).append(
                {"source": source, "raw_value": deepcopy(raw_fields[key])})
            if key in _PATHS:
                path = (base / value).resolve()
                resource_ids[key] = _relative_id(path, root)
                value = str(path)
            values[key] = value

    def visit(path):
        path = path.resolve()
        source = _relative_id(path, root)
        if path in active:
            chain = " -> ".join(_relative_id(item, root) for item in active + [path])
            raise ConfigError(f"Circular extends: {chain}")
        try:
            document = yaml.load(path.read_text(encoding="utf-8"), Loader=_StrictLoader)
            fields = _fragment(document)
        except (OSError, UnicodeError, yaml.YAMLError, ConfigError) as error:
            raise ConfigError(f"{source}: {error}") from error
        parents = document.get("extends", [])
        if not isinstance(parents, list) or any(
                not isinstance(item, str) or not item.strip() or "$" in item for item in parents):
            raise ConfigError(f"{source}: extends must be a list of nonempty file paths")
        active.append(path)
        try:
            for parent in parents:
                visit(path.parent / parent)
            raw_fields = {key: (document[key] if "." not in key else
                          document[key.split(".")[0]][key.split(".")[1]]) for key in fields}
            apply(fields, raw_fields, source, path.parent)
            load_order.append(source)
        finally:
            active.pop()

    visit(root / config_path)
    if cli_overrides is not None:
        if not isinstance(cli_overrides, dict):
            raise ConfigError("CLI overrides must be a mapping of dotted field names")
        typed = {key: _validate(key, value) for key, value in cli_overrides.items()}
        apply(typed, cli_overrides, "<CLI>", root)
    missing = sorted(set(_FIELDS) - {'pairing.window'} - set(values))
    if missing:
        raise ConfigError("Missing required fields: " + ", ".join(missing))
    if values['pairing.strategy'] == 'sequential_window':
        if 'pairing.window' not in values:
            raise ConfigError('sequential_window requires pairing.window')
    elif 'pairing.window' in values:
        raise ConfigError('pairing.window is only valid for sequential_window')
    for key in ("scene.manifest", "backend.weights_manifest"):
        if not Path(values[key]).is_file():
            raise ConfigError(f"Resource file does not exist: {key} = {values[key]}")
    if Path(values["output.root"]).exists() and not Path(values["output.root"]).is_dir():
        raise ConfigError("output.root exists but is not a directory")
    _check_device(values["runtime.device"])
    portable = {key: resource_ids.get(key, value) for key, value in values.items()}
    canonical = json.dumps(_nested(portable), sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False, allow_nan=False)
    semantic_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return ResolvedConfig(_nested(values), sources, resource_ids, semantic_hash, load_order)
