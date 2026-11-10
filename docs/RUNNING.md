# Configured reconstruction entry: normal pipeline binding

`scripts/run_reconstruction.py` resolves configuration and calls
`ReconstructionPipeline.run(resolved_config)`. `scripts/validate_scene_01.py` is preserved as the baseline reference.
Own ImageRecord/ImageBatch loading is connected; see docs/IMAGE_LOADING.md.
Own PairGraph and its directed bridge are connected; see docs/PAIRING.md.
The own sparse backend lifecycle and owned CPU result extraction are connected;
see docs/BACKEND.md. Standard results, exporter and staged pipeline are connected;
see docs/RESULTS.md, docs/EXPORTING.md and docs/PIPELINE.md. This code binding has
CPU fixture validation; real GPU execution through this new binding remains unverified.

Preflight (does not load a model or import the reconstruction backend):

```bash
python scripts/run_reconstruction.py --config configs/scenes/scene_01.yaml --dry-run
```

Actual reconstruction, after preflight has passed:

```bash
python scripts/run_reconstruction.py --config configs/scenes/scene_01.yaml
```

Scalar overrides are explicit, typed, and recorded. Paths in CLI overrides use the
repository root. Duplicate overrides, containers and executable YAML tags are rejected.
For example, a separate parameter experiment can use
`--set backend.coarse_iterations=80`; the default baseline is unchanged.

The manifest controls image order. Images and the actual checkpoint are hashed before
model loading. Weight and notice paths are relative to assets/weights.json. Existing
source declarations are retained as declarations, not independently verified provenance.
Each invocation creates a new outputs/<scene_id>/<UTC_timestamp>_<random_id> directory.
Configuration evidence, asset evidence and run_status.json are written before inference.

Actual runs preserve gradients for sparse-GA optimization and use FP32 model parameters
and the configured iteration/lr/confidence/intrinsic values. Pairing uses the validated
complete, sequential_window or explicit graph and its fixed directed ordering. The existing
scene_01 configuration remains complete; sequential_window requires declared capture order,
and explicit requires manifest edges. See docs/PAIRING.md for declaration and validation rules.
Vendor imports must come from the pinned repository snapshot. Backend preprocessing and
numerical behavior remain those of that snapshot; no new determinism guarantee is made.

Per-run sparse-GA workspace is temporary and is removed after result extraction, including
on failure. Optimizer stdout is retained in reconstruction_optimizer.log. Tracebacks and
progress bars written to stderr remain terminal output; this is not a full log collector.

Normal output uses `artifacts/scene.ply`, `artifacts/cameras.json`,
`artifacts/point_sources.npz` and `artifacts/artifacts.json`. Exporter readback checks
their correspondence before publication. Points/cameras share the first registered
camera reference transform; scale stays unverified. `standard_result.json` records
masking and availability. `reconstruction.json` links standard result/artifact metadata
and observed backend losses/resources. It replaces the legacy camera-array layout.
The baseline script and historical outputs retain their original layout. This normal
M2 entry does not yet generate a new preview image. A `succeeded` status proves the
declared phases and cleanup passed, not geometry quality. Backend time excludes
model loading, preflight hashing, result extraction and export.
The backend timer stops after alignment and device synchronization, before writing
the captured stdout log and extracting owned CPU arrays. Earlier entry timings
also included captured-log write/print overhead; this is not a performance acceptance.
This step does not freeze performance/quality thresholds or implement cross-run cache,
full-scene matching statistics or standalone quality/report modules. Source-pixel capture
is connected in code, verified with CPU fixtures and accepted for the separate real-GPU
two-image smoke described below. Other scenes and GPU strategies remain unverified.

Each actual run writes pair_graph.json before model loading and links it through
pair_graph_hash and logical_edge_count in reconstruction.json. The final manifest
also indexes immutable per-run evidence by SHA256 and byte count.
The graph retains image identities, unique undirected edges, stable pair IDs,
generation basis, connectivity and directed ordering; no model forward count is inferred.
For the independent pairing check and entry preflight against its saved evidence,
see docs/PAIRING.md. A passed pairing preflight has not performed reconstruction.

The previously verified model-free backend boundary check remains available:

```bash
python scripts/check_backend_entry.py --config configs/scenes/scene_01.yaml --reference work/backend_io_checks/<check_id>/backend_io_check.json
```

This verifies resources/weights, source files, image identities and parameters;
it does not import MASt3R or validate effective model dtype and scene extraction.
Actual runs load a verified local FP32 model, set eval and freeze its parameters.
Geometry optimization explicitly enables gradients and uses an isolated temporary
workspace. Owned CPU arrays are extracted before cache cleanup and exported after
backend closure. backend_fingerprint.json records observed model/device/source
identity; backend_result.json records observed output dtypes, counts and unavailable
metrics. backend_events.jsonl links coarse execution events to run/config identity;
these are stage events, not counts of model forwards or per-iteration instrumentation.

Run CPU contracts:

```bash
python -m unittest discover -s tests/unit -p 'test_*.py' -v
```

New normal pipeline CPU fixture check (does not read old GPU artifacts):

```bash
python scripts/check_normal_pipeline.py
```

The check runs normal CPU preflight/image loading/pairing/normalization/export and
readback with a synthetic backend and non-loadable checkpoint. Its two cameras and
three observations prove orchestration only. Reports/logs stay under a fresh
`work/normal_pipeline_checks/<UTC+ID>/` directory. Real GPU pipeline, complete M2,
source pixels and geometry acceptance remain false in that historical fixture report.

Subsequent CPU checks have separate evidence: scripts/check_sparse_pixels.py validates
the normal-entry capture and export of anchor xy; scripts/check_pairing_strategies.py
validates four strategy configurations and two rejected declarations. Their stage records
are benchmarks/scene_01/sparse_pixels_cpu_verified.json and
benchmarks/scene_01/pairing_strategies_cpu_verified.json. These fixtures do not load the
actual model or verify GPU matching/optimization, GPU strategy quality or GPU source pixels.

The backend captures processed-image x,y at the pinned SparseGA constructor and passes
owned CPU copies through normalization to artifacts/point_sources.npz. Source-image and
observation indices, points, colors and xy use the same mask; duplicate xy rows are retained.
Partial per-image xy is rejected. Historical inputs without xy remain unavailable, and
reconstruction point confidence remains unavailable. See docs/RESULTS.md and docs/PIPELINE.md.

The original scene_01 manifest still contains seven images. A separate two-image
normal-entry smoke is now declared by configs/scenes/scene_01_smoke.yaml and
data/scene_01/smoke_01_02_manifest.json. It references the original 01.png / 02.png
without copying or changing them, retains their source declarations, uses complete
pairing and inherits the existing base/T4 parameters. Its scene ID is scene_01_smoke,
so each normal invocation writes a fresh outputs/scene_01_smoke/<UTC+ID>/ directory.

The first smoke --dry-run passed configuration and asset_validation only, with two
images and two expected directed items. Its run is
outputs/scene_01_smoke/20261009T193347Z_9cc676f5f838, executed at
1f2cc983cf4ddc87a3f182d203f73ca66dade7bf before the smoke assets were committed.
The original run metadata is retained. Preflight did not decode images, prepare a
backend, load the model, perform matching/optimization or export a reconstruction.

The commands above are usage examples. Existing GPU probe and seven-image runs are
historical evidence; they do not verify the later normal-entry/xy binding. The first
smoke preflight remains resource-validation evidence only. Its execution commit and
original false acceptance fields are preserved.

A subsequent real-GPU normal-entry smoke completed at commit
6203956b61209f93b72be450c207dcb3cf198465, with run
outputs/scene_01_smoke/20261010T070434Z_87a89eca26d2. Its consolidated acceptance record is
benchmarks/scene_01_smoke/normal_entry_gpu_smoke_verified.json. This one two-image
complete-pairing run verifies the actual normal pipeline on Tesla T4 / cuda:0 / fp32,
constructor anchor xy capture, source-row/color/mask/camera correspondence, export
readback and temporary-resource cleanup. All ten phases succeeded; all 19,848 exported
observation rows matched their source-pixel colors. Duplicate xy rows are retained;
observation count is not a unique physical point count.

Acceptance was recorded separately after metadata and artifact checks. The original
run, preflight and historical CPU/GPU reports are unchanged, including their false
acceptance fields. The artifact check reused the pinned CPU image loader to reproduce
the recorded pixel/tensor hashes; it did not rerun the model or optimizer or provide
an independent preprocessing implementation. Live anchors were not persisted for a
second independent extraction; source verification relies on the pinned constructor
capture and the verified row/export chain.

Other GPU pairing strategies, actual model forward count, full-scene matching metrics
and reconstruction confidence are not accepted by this smoke. Complete M2, repeated
cold runs and fresh-instance deployment gates, geometry quality, physical scale and
Golden freeze remain pending.
