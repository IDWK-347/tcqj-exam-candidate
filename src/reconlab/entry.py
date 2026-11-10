"""Configured CLI and shared preflight/bridges; normal runs use ReconstructionPipeline."""
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid
from types import SimpleNamespace

import yaml

from .config import ConfigError, resolve_config
from .pairing import build_pair_graph, to_directed_pairs


class EntryError(ValueError):
    """Invalid inputs or an unsuccessful reconstruction/output check."""


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise EntryError(f"Duplicate JSON key: {key} in {path}")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _expected_hash(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise EntryError(f"Invalid SHA256 declaration: {label}")
    return value.lower()


def _asset_path(base, raw, scope):
    if not isinstance(raw, str) or not raw.strip() or Path(raw).is_absolute():
        raise EntryError(f"Asset path must be a nonempty relative path: {raw!r}")
    path = (base / raw).resolve()
    if not path.is_relative_to(scope):
        raise EntryError(f"Asset path escapes its permitted directory: {raw}")
    if not path.is_file():
        raise EntryError(f"Asset file does not exist: {path}")
    return path


def parse_overrides(values):
    """Parse scalar FIELD=VALUE options with safe YAML, never eval/exec."""
    result = {}
    for item in values:
        key, separator, raw = item.partition("=")
        if not separator or not key.strip() or key in result:
            raise EntryError(f"Invalid or duplicate --set: {item!r}")
        try:
            value = yaml.safe_load(raw)
        except yaml.YAMLError as error:
            raise EntryError(f"Invalid --set value for {key}: {error}") from error
        if type(value) not in (str, int, float, bool):
            raise EntryError(f"--set only accepts scalar strings, numbers and booleans: {key}")
        result[key] = value
    return result


@dataclass(frozen=True)
class PreparedRun:
    root: Path
    config: object
    output: Path
    files: list
    weight: Path
    settings: dict
    evidence: dict


def validate_assets(config, *, project_root, output, run_id):
    """Shared resource validation; directory/status lifecycle belongs to caller."""
    root = Path(project_root).resolve()
    output = Path(output)
    values = config.values
    manifest_path = Path(values["scene"]["manifest"])
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("images"), list):
        raise EntryError("Scene manifest must contain an images list")
    if len(manifest["images"]) < 2:
        raise EntryError("Reconstruction requires at least two images")
    source = manifest.get("source")
    if not isinstance(source, str) or not source.strip():
        raise EntryError("Scene manifest must declare its source")
    files, inputs, seen = [], [], set()
    for entry in manifest["images"]:
        if not isinstance(entry, dict):
            raise EntryError("Each manifest image must be a mapping")
        path = _asset_path(manifest_path.parent, entry.get("file"), manifest_path.parent)
        if path in seen:
            raise EntryError(f"Duplicate manifest image: {entry['file']}")
        seen.add(path)
        expected = _expected_hash(entry.get("sha256"), entry["file"])
        actual = _hash(path)
        if actual != expected:
            raise EntryError(f"Image SHA256 mismatch: {entry['file']}")
        files.append(path)
        inputs.append({"file": entry["file"], "sha256": actual})
    declaration = manifest.get('pairing', {})
    from .pairing import validate_pairing_declaration
    image_ids = tuple(item.get('image_id', item['file']) for item in manifest['images'])
    declaration = validate_pairing_declaration(declaration, image_ids)
    expected_graph = build_pair_graph(SimpleNamespace(
        records=[SimpleNamespace(image_id=value, manifest_index=index)
                 for index, value in enumerate(image_ids)], pairing_declaration=declaration),
        values['pairing'])
    weights_path = Path(values["backend"]["weights_manifest"])
    weights = _read_json(weights_path)
    if (not isinstance(weights, dict) or type(weights.get("schema_version")) is not int
            or weights["schema_version"] != 1):
        raise EntryError("Weights manifest must declare integer schema_version: 1")
    weight = _asset_path(weights_path.parent, weights.get("file"), root)
    notice = _asset_path(weights_path.parent, weights.get("license_notice"), root)
    expected_weight = _expected_hash(weights.get("sha256"), "checkpoint")
    print("校验实际权重文件 SHA256……", flush=True)
    weight_sha = _hash(weight)
    if weight_sha != expected_weight:
        raise EntryError("Checkpoint SHA256 mismatch")
    backend = values["backend"]
    settings = {"subsample": backend["subsample"], "device": values["runtime"]["device"],
                "lr1": backend["coarse_lr"], "niter1": backend["coarse_iterations"],
                "lr2": backend["refine_lr"], "niter2": backend["refine_iterations"],
                "matching_conf_thr": backend["matching_conf_thr"],
                "shared_intrinsics": backend["shared_intrinsics"]}
    evidence = {"schema_version": 1, "check": "configured_reconstruction_preflight",
        "run_id": run_id, "scene_id": values["scene"]["id"],
        "config_semantic_hash": config.semantic_hash,
        "sources": _read_json(root / "vendor/manifest.json"),
        "manifest_sha256": _hash(manifest_path), "weight_sha256": weight_sha,
        "weight_resource": weight.relative_to(root).as_posix(),
        "license_notice": notice.relative_to(root).as_posix(),
        "input_provenance": source, "inputs": inputs,
        "image_count": len(files), "expected_directed_pair_count": 2 * len(expected_graph.edges),
        "expected_pair_graph_hash": expected_graph.graph_hash,
        "image_size": values["preprocess"]["long_edge"],
        "square_ok": values["preprocess"]["square_ok"], "settings": settings}
    _write_json(output / "preflight.json", evidence)
    return PreparedRun(root, config, output, files, weight, settings, evidence)


def prepare_run(config_path, overrides=None, *, project_root):
    """Legacy model-free checks/probe preflight; normal CLI uses the pipeline."""
    root = Path(project_root).resolve()
    config = resolve_config(config_path, overrides, project_root=root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:12]
    output = Path(config.values["output"]["root"]) / config.values["scene"]["id"] / run_id
    output.mkdir(parents=True, exist_ok=False)
    print("本次输出目录：", output, flush=True)
    try:
        config.write_evidence(output)
        _write_json(output / "run_status.json", {"schema_version": 1,
                    "status": "validating_assets", "run_id": run_id})
        run = validate_assets(config, project_root=root, output=output, run_id=run_id)
        _write_json(output / "run_status.json", {"schema_version": 1,
                    "status": "prepared", "run_id": run_id})
        return run
    except Exception as error:
        _write_json(output / "run_status.json", {"schema_version": 1, "status": "failed",
                    "phase": "asset_validation", "run_id": run_id,
                    "error_type": type(error).__name__, "error": str(error)})
        raise


def _build_run_pairs(run, batch, images):
    """Verify prepared identities and persist the graph before model loading."""
    if ([record.path for record in batch.records] != run.files
            or batch.manifest_sha256 != run.evidence["manifest_sha256"]
            or [record.file_sha256 for record in batch.records]
            != [item["sha256"] for item in run.evidence["inputs"]]):
        raise EntryError("Image batch differs from validated preflight inputs")
    graph = build_pair_graph(batch, run.config.values["pairing"])
    pairs = to_directed_pairs(graph, images,
                              image_ids=[record.image_id for record in batch.records])
    if len(pairs) != run.evidence["expected_directed_pair_count"]:
        raise EntryError("Strategy directed pair count differs from preflight expectation")
    expected_graph_hash = run.evidence.get('expected_pair_graph_hash')
    if expected_graph_hash is not None and graph.graph_hash != expected_graph_hash:
        raise EntryError('Pair graph differs from preflight strategy/declaration')
    _write_json(run.output / "pair_graph.json", {"schema_version": 1,
        "run_id": run.evidence["run_id"], "config_semantic_hash": run.config.semantic_hash,
        "manifest_sha256": batch.manifest_sha256,
        "processed_pixel_hash": batch.processed_pixel_hash, "pair_graph": graph.metadata()})
    return graph, pairs



def _backend_options(run):
    from .backends.mast3r import ReconstructionOptions
    fields = ('subsample', 'coarse_lr', 'coarse_iterations', 'refine_lr',
              'refine_iterations', 'matching_conf_thr', 'shared_intrinsics')
    options = ReconstructionOptions(**{field: run.config.values['backend'][field] for field in fields})
    if options.upstream_kwargs(run.config.values['runtime']['device']) != run.settings:
        raise EntryError('Backend parameter mapping differs from validated preflight settings')
    return options


def _backend_specs(run):
    from .backends.mast3r import RuntimeSpec, WeightsSpec
    return (WeightsSpec(run.weight, run.evidence['weight_sha256']),
            RuntimeSpec(run.config.values['runtime']['device'], run.config.values['runtime']['precision']))


def _load_run_images(run):
    from .image_loader import load_scene
    batch = load_scene(run.config.values['scene']['manifest'], run.config.values['preprocess'])
    if ([record.path for record in batch.records] != run.files
            or batch.manifest_sha256 != run.evidence['manifest_sha256']
            or [record.file_sha256 for record in batch.records]
            != [item['sha256'] for item in run.evidence['inputs']]):
        raise EntryError('Image batch differs from validated preflight inputs')
    batch.write_evidence(run.output)
    return batch


def _build_backend_inputs(run, batch):
    from .backends.sparse_io import prepare_vendor_inputs
    graph = build_pair_graph(batch, run.config.values['pairing'])
    inputs = prepare_vendor_inputs(batch, graph)
    checked_graph, checked_pairs = _build_run_pairs(run, batch, inputs.images)
    if checked_graph != graph or not all(a is c and b is d
            for (a, b), (c, d) in zip(checked_pairs, inputs.pairs)):
        raise EntryError('Backend input bridge differs from the verified entry pairing boundary')
    return graph, inputs, _backend_options(run)


def _prepare_backend_inputs(run):
    """Preserve the verified model-free checker/probe boundary."""
    batch = _load_run_images(run)
    graph, inputs, options = _build_backend_inputs(run, batch)
    return batch, graph, inputs, options


def _append_backend_event(run, event):
    target = run.output / 'backend_events.jsonl'
    if not target.resolve().is_relative_to(run.output.resolve()):
        raise EntryError('Backend event file is outside this run')
    record = {**event, 'schema_version': 1, 'run_id': run.evidence['run_id'],
              'config_semantic_hash': run.config.semantic_hash,
              'recorded_at_utc': datetime.now(timezone.utc).isoformat()}
    with target.open('a', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def main(argv=None, *, project_root):
    parser = argparse.ArgumentParser(description="配置驱动的稀疏重建入口")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true", help="只校验资源与参数，不加载模型")
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args(argv)
    try:
        from .pipeline import ReconstructionPipeline
        config = resolve_config(args.config, parse_overrides(args.set), project_root=project_root)
        ReconstructionPipeline(project_root, dry_run=args.dry_run).run(config)
        return 0
    except (ConfigError, EntryError, OSError, ValueError, KeyError, RuntimeError, ImportError) as error:
        print(f"执行失败：{error}", file=sys.stderr)
        return 1
