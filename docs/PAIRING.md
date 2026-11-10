# Declared pairing and entry integration

`reconlab.pairing.build_pair_graph(images, pairing_config)` consumes ImageBatch
identities and manifest indices without importing the image loader, torch or a
backend. The three supported strategies are complete, sequential_window and
explicit. Strategy-specific fields and manifest declarations are strictly checked.
The reconstruction entry now uses the own graph and directed bridge. It no longer
calls pinned make_pairs. The independent checker retains that source as a reference.
Passing CPU gates is not evidence of an integrated GPU run.

## Strategy declarations

The existing complete behavior, pair IDs, forward/reverse ordering and graph hash
remain unchanged. Old manifests need no new fields and retain complete pairing.
Optional scene-manifest `pairing` contains only `order_semantics` and
`explicit_edges`; explicit null, unknown keys and invalid declarations are rejected.
ImageBatch owns a copied `pairing_declaration`, and image_preprocessing.json records
it when present. A list's storage order or numeric filenames never imply sequence.

- `{"strategy": "sequential_window", "window": W}` requires manifest
  `"pairing": {"order_semantics": "capture_sequence"}`. W is the maximum positive
  manifest-index distance between endpoints; no wraparound is performed. W >= N-1
  covers every pair but retains its declared value and strategy identity.
- `{"strategy": "explicit"}` requires manifest
  `"pairing": {"explicit_edges": [["id_a", "id_b"], ...]}`. Those edges alone form
  the graph; no automatic edge supplementation occurs. Unknown IDs, self loops,
  malformed pairs and repeated/reverse logical edges fail. Input orientation/order
  is normalized into manifest traversal order; IDs are never sorted as image order.
- `order_semantics: "unordered"` explicitly declares no capture sequence. It is
  compatible with complete/explicit but cannot enable sequential_window.

All graphs must cover their manifest nodes and be connected. Disconnection errors
retain components in manifest order. Each new strategy records its generation
basis and uses `manifest_forward_block_then_reverse_v1`; forward endpoints follow
descending manifest indices, followed by the complete reversed block. Window
parameters/sequence declaration are part of its graph hash. Pair ID remains an
undirected identity shared across strategies; graph hash remains distinct from
visual-overlap or geometry-quality evidence.

PairGraph has schema_version=1, an ordered node tuple, canonical undirected edges,
stable pair IDs, strategy parameters, generation basis, connected components and
a graph hash. Each edge's endpoints are sorted by image ID for unambiguous identity;
the graph's node order always remains manifest order. Pair IDs are `pair_` followed
by SHA256 of the compact UTF-8 JSON array of its two canonical IDs. They do not
depend on image filenames being numeric, separator concatenation or machine paths.
When explicit IDs are absent, the existing image loader uses each file string.

The graph hash covers schema, ordered nodes, ordered canonical edges (including
pair IDs and generation basis), strategy, symmetrization and directed order policy.
It is a topology/order identity, not an image-content hash, run fingerprint,
quality score or cache key. Reordering the manifest changes the graph hash but
does not change the ID of an unchanged logical pair.

Validation rejects insufficient or repeated nodes, self loops, unknown endpoints,
duplicate undirected edges, wrong IDs/basis, disconnected components and incomplete
complete graphs. Disconnection errors expose components in manifest order. It also
checks edge order; checking only the count is insufficient. Structural connectivity
does not prove visual overlap or accepted reconstruction quality.

`to_directed_pairs(graph, backend_images, *, image_ids)` revalidates the graph,
requires exact image identity/order and idx agreement, and returns existing backend
dictionaries by reference. No pixels are copied or processed. The fixed policy
matches the pinned vendor complete branch: for each increasing i, visit j<i; then
append the entire reverse block. Three views give `(1,0),(2,0),(2,1),(0,1),(0,2),(1,2)`.
Seven views give 21 logical edges and 42 directed items. Neither number claims an
observed model forward count; that metric remains unavailable in graph evidence.

Run the independent CPU contracts:

```bash
python -m unittest discover -s tests/unit -p 'test_*.py' -v
```

Check actual scene identities, tensors and directed object order without a model:

```bash
python scripts/check_pairing.py --config configs/scenes/scene_01.yaml
```

This historical checker compares the complete branch only. It loads the scene once via the existing loader and makes the same temporary
backend dictionaries as entry.py. It compares every returned object and its order
against pinned image_pairs.py, whose SHA256 must be
`b0b28df0868c0663b1dccb0396391efe7b9b0711fab09c1bb57fd129847310a5`.
This source is used as a test reference, not modified or copied into pairing.py.
The default JSON evidence is saved in a fresh work/pairing_checks/<UTC+ID> directory.
`--output` may name a new JSON file inside the repository; existing evidence is
never overwritten. Evidence retains configuration, manifest and processed-pixel
hashes, graph/directed-item mapping and unavailable/unverified metric states.
No checkpoint is loaded, no optimization runs and no GPU result is asserted.

The entry's `_build_run_pairs` boundary rechecks image paths/order, manifest hash and
image byte hashes against prepared inputs, builds the graph, checks the directed
count and writes pair_graph.json before model loading. That file links run_id,
configuration, manifest and processed-pixel hashes to the full graph. Reconstruction
reports add logical_edge_count, pair_graph_hash and pair_graph_file; existing
directed_pair_count remains the number passed to the backend, not observed forwards.

`scripts/check_pairing_entry.py --config configs/scenes/scene_01.yaml --reference PATH`
performs real asset preflight (including checkpoint SHA256), loads actual CPU image
tensors and invokes the same entry boundary. PATH names a successful saved pairing
check JSON inside the repository. It compares input hashes, graph metadata and the
actual directed sequence against that reference, then writes pairing_entry_check.json
in a fresh normal run directory. It does not import or load the reconstruction model.
This check has a distinct status of pairing_preflight_passed, not completed reconstruction.
Failure retains run_status.json when asset preflight produced a run.

The entry preflight now derives expected directed-item count and graph hash from
manifest IDs/declarations and the effective strategy. The loaded batch must produce
that same graph, including when two graphs have equal counts. It no longer applies
N(N-1) to every strategy. Missing sequence semantics and disconnected explicit
graphs fail in asset_validation before backend preparation; the normal pipeline
keeps a failed final manifest.

`python scripts/check_pairing_strategies.py` creates new synthetic fixtures in
work/pairing_strategy_checks. Four successful cases run the actual normal pipeline,
normalization/export/readback with an explicit synthetic backend: complete (12
directed items), window1 (6), window2 (10), explicit (6). Two rejected cases retain
failure manifests. The four-camera/four-observation outputs do not come from
MASt3R or optimization. No actual model/GPU, old reconstruction replay, or changes
to scene_01 are involved. GPU strategy behavior, complete M2, geometry/resource
acceptance and Golden freeze remain unverified.
