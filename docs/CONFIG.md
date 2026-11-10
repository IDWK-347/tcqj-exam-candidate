# Current configuration contract

This step implements the fields present in the verified scene_01 configuration.
The incremental scripts/run_reconstruction.py entry now consumes this configuration;
see docs/RUNNING.md for preflight and actual reconstruction.

- `resolve_config(config_path, cli_overrides=None, *, project_root)` returns
  `ResolvedConfig(values, sources, resource_ids, semantic_hash, load_order)`.
- Relative config entry paths and typed CLI override paths use `project_root`.
  Each YAML resource and `extends` path uses its own declaring file's directory.
- Parents are applied in list order, then the current file, then dotted typed
  overrides. Nested section fields merge individually. Repeated ancestry is
  replayed in this order; only an active recursion cycle is an error.
- Every YAML file requires integer `schema_version: 1`. Duplicate keys, aliases,
  unknown fields, invalid types, and missing final fields fail explicitly.
- Only the current backend, declared complete/sequential_window/explicit pairing, 512-pixel preprocessing and FP32
  precision are supported. Integers must be positive, learning rates positive
  and finite, matching confidence nonnegative and finite, booleans actual booleans.
  These validity checks are not a measured GPU cost or quality budget.
- `cpu`, `cuda`, and `cuda:<index>` are explicit device requests. A CUDA request
  checks availability and the index before returning. It never falls back to CPU.
  This check does not execute kernels or prove model compatibility.
- Current configuration and resource paths must stay inside `project_root`.
  Absolute paths inside it are accepted. External mounts and environment variable
  substitution are not yet supported. Output root may be absent but must not be
  an existing file. Scene and weights manifests must exist as files.
- The resource IDs and semantic hash use repository-relative POSIX paths, not
  machine roots or the working directory. This is a configuration hash, not an
  asset content hash, run fingerprint or cache key.
- Each field's source history records raw values in application order. Resolved
  runtime paths are absolute. `write_evidence(directory)` writes
  `resolved_config.yaml` and `config_sources.json` before model execution.
  Values must not be changed after resolution; resolve again for another run.
- `scripts/check_config.py` checks the current configuration against recorded
  scene_01 parameters and the declared weight hash, then writes evidence. It does
  not load the model, hash the actual checkpoint, or perform reconstruction.

The incremental entry supports restricted scalar `--set` parsing. Deferred design
features include allowlisted environment substitution, general list-valued fields,
cache/quality/export settings, resource budgets and the final module decomposition.
Unsupported options fail instead of being ignored.

Pairing now supports three final configurations: `{"strategy": "complete"}`,
`{"strategy": "sequential_window", "window": 1}` and `{"strategy": "explicit"}`.
`window` must be a positive integer (booleans rejected), is required only for
sequential_window, and is forbidden for complete/explicit. This check runs after
inheritance and CLI merging; a window inherited from a parent is not silently
discarded when a child selects another strategy. The value and source history
participate in the configuration semantic hash.

Sequence semantics and explicit edges are read from the scene manifest, not CLI
lists. See docs/PAIRING.md. A requested strategy requires its declaration and a
connected graph during preflight, before model preparation. No sequence semantics
have been added to the existing scene_01 manifest or default configuration.

Run CPU contracts from the repository root:

```bash
python -m unittest discover -s tests/unit -p test_config.py -v
python scripts/check_config.py
```
