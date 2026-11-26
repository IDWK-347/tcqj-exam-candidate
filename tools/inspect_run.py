#!/usr/bin/env python3
"""只读汇总一个 run 目录，供 BASELINE / DIAGNOSIS / EXPERIMENTS 取证。

用法：
    python3 tools/inspect_run.py outputs/scene_a/<run_id>
    python3 tools/inspect_run.py --latest outputs/scene_a
    python3 tools/inspect_run.py outputs/scene_a/<run_id> --json

设计约束：
- 只读，不写任何文件；
- 不依赖项目内部模块，避免影响 run 的 implementation_sources 指纹；
- 文件缺失时降级输出（失败的 run 往往只有部分产物），不抛异常。
"""

import argparse
import json
import sys
from pathlib import Path


def _load(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _dur_seconds(a, b):
    """ISO8601 时间差，失败返回 None。"""
    from datetime import datetime
    try:
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds()
    except (TypeError, ValueError):
        return None


def _fmt(value, nd=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def summarize(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    manifest = _load(run_dir / "run_manifest.json") or {}
    status = _load(run_dir / "run_status.json") or {}
    standard = _load(run_dir / "standard_result.json") or {}
    recon = _load(run_dir / "reconstruction.json") or {}
    pair = _load(run_dir / "pair_graph.json") or {}
    artifacts = _load(run_dir / "artifacts" / "artifacts.json") or {}
    preflight = _load(run_dir / "preflight.json") or {}

    out = {
        "run_dir": str(run_dir),
        "run_id": manifest.get("run_id", run_dir.name),
        "scene_id": manifest.get("scene_id"),
        "mode": manifest.get("mode"),
        "status": manifest.get("status", status.get("status")),
        "created_at_utc": manifest.get("created_at_utc"),
        "ended_at_utc": manifest.get("ended_at_utc"),
        "error": manifest.get("error"),
        "phases": [],
        "pairing": None,
        "result": None,
        "artifacts": None,
        "backend": None,
    }

    for phase in manifest.get("phases", []):
        out["phases"].append({
            "name": phase.get("name"),
            "status": phase.get("status"),
            "seconds": _dur_seconds(phase.get("started_at_utc"), phase.get("ended_at_utc")),
            "error": phase.get("error"),
        })

    graph = pair.get("pair_graph") or {}
    if graph:
        out["pairing"] = {
            "strategy": graph.get("strategy_parameters", {}).get("strategy"),
            "nodes": len(graph.get("nodes", [])),
            "logical_edges": graph.get("logical_edge_count"),
            "directed_items": graph.get("directed_item_count"),
            "components": graph.get("connected_components"),
            "graph_hash": graph.get("graph_hash"),
        }

    if standard:
        avail = standard.get("availability", {})
        out["result"] = {
            "point_count": standard.get("point_count"),
            "camera_count": standard.get("camera_count"),
            "scale_status": standard.get("scale_status"),
            "point_confidence": (avail.get("point_confidence") or {}).get("status"),
            "source_pixels_xy": (avail.get("source_pixels_xy") or {}).get("status"),
            "processed_hw": standard.get("processed_hw"),
            "retained_count": (standard.get("filtering") or {}).get("retained_count"),
            "per_view": (standard.get("filtering") or {}).get("per_view"),
        }

    if artifacts:
        out["artifacts"] = {
            "point_count": artifacts.get("point_count"),
            "camera_count": artifacts.get("camera_count"),
            "files": {name: meta.get("bytes") for name, meta in (artifacts.get("artifacts") or {}).items()},
            "readback_ok": all((artifacts.get("readback_checks") or {}).values())
            if artifacts.get("readback_checks") else None,
        }

    if recon:
        out["backend"] = {
            "seconds_excluding_model_load": recon.get("reconstruction_seconds_excluding_model_load"),
            "peak_allocated_gib": recon.get("peak_allocated_gib"),
            "final_losses": recon.get("final_losses"),
            "per_image_observation_counts": recon.get("per_image_observation_counts"),
            "measurement_scope": recon.get("backend_measurement_scope"),
            "geometry_quality_accepted": recon.get("geometry_quality_accepted"),
        }

    if preflight:
        out["preflight"] = {
            "image_count": preflight.get("image_count"),
            "expected_directed_pair_count": preflight.get("expected_directed_pair_count"),
            "weight_sha256": (preflight.get("weight_sha256") or "")[:16],
            "manifest_sha256": (preflight.get("manifest_sha256") or "")[:16],
        }

    out["has"] = sorted(p.name for p in run_dir.iterdir() if p.is_file()) if run_dir.is_dir() else []
    return out


def print_human(summary: dict) -> None:
    print(f"run_dir : {summary['run_dir']}")
    print(f"run_id  : {summary['run_id']}")
    print(f"scene   : {summary['scene_id']}   mode={summary['mode']}   status={summary['status']}")
    if summary.get("error"):
        err = summary["error"]
        print(f"error   : [{err.get('phase')}] {err.get('type')}: {err.get('message')}")

    if summary.get("preflight"):
        pf = summary["preflight"]
        print(f"preflight: images={pf['image_count']}  directed_pairs={pf['expected_directed_pair_count']}  "
              f"weight_sha={pf['weight_sha256']}…")

    if summary["phases"]:
        print("\n-- phases --")
        for p in summary["phases"]:
            secs = "-" if p["seconds"] is None else f"{p['seconds']:.2f}s"
            print(f"  {p['name']:<22s} {p['status']:<10s} {secs:>9s}")
        total = sum(p["seconds"] for p in summary["phases"] if p["seconds"])
        print(f"  {'TOTAL':<22s} {'':<10s} {total:>8.2f}s")

    if summary["pairing"]:
        g = summary["pairing"]
        print("\n-- pairing --")
        print(f"  strategy={g['strategy']} nodes={g['nodes']} logical_edges={g['logical_edges']} "
              f"directed={g['directed_items']}")
        print(f"  components={g['components']}")
        print(f"  graph_hash={g['graph_hash']}")

    if summary["result"]:
        r = summary["result"]
        print("\n-- standard_result --")
        print(f"  points={r['point_count']} cameras={r['camera_count']} scale_status={r['scale_status']}")
        print(f"  point_confidence={r['point_confidence']} source_pixels_xy={r['source_pixels_xy']}")
        print(f"  processed_hw={r['processed_hw']}")
        for view in r.get("per_view") or []:
            print(f"    {view.get('image_id')}: input={view.get('input_count')} retained={view.get('retained_count')}")

    if summary["backend"]:
        b = summary["backend"]
        print("\n-- reconstruction --")
        print(f"  alignment_seconds={_fmt(b['seconds_excluding_model_load'])}  "
              f"peak_allocated_gib={_fmt(b['peak_allocated_gib'])}")
        print(f"  final_losses={b['final_losses']}")
        print(f"  per_image_observation_counts={b['per_image_observation_counts']}")
        print(f"  geometry_quality_accepted={b['geometry_quality_accepted']}")

    if summary["artifacts"]:
        a = summary["artifacts"]
        print("\n-- artifacts --")
        print(f"  points={a['point_count']} cameras={a['camera_count']} readback_ok={a['readback_ok']}")
        for name, size in (a["files"] or {}).items():
            print(f"    {name:<20s} {size} bytes")

    missing = [name for name in ("standard_result.json", "reconstruction.json",
                                 "pair_graph.json", "artifacts/artifacts.json")
               if not (Path(summary["run_dir"]) / name).exists()]
    if missing:
        print(f"\n缺失产物: {', '.join(missing)}")


def latest_run(scene_root: Path) -> Path:
    runs = [p for p in scene_root.iterdir() if p.is_dir()]
    if not runs:
        raise SystemExit(f"没有 run 目录: {scene_root}")
    return sorted(runs, key=lambda p: p.name)[-1]


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读汇总一个 run 目录")
    parser.add_argument("run_dir", help="run 目录，或配合 --latest 传场景目录")
    parser.add_argument("--latest", action="store_true", help="把参数当作场景目录，取字典序最新的一次 run")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    target = Path(args.run_dir)
    if args.latest:
        target = latest_run(target)
    if not target.is_dir():
        raise SystemExit(f"不是目录: {target}")

    summary = summarize(target)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
