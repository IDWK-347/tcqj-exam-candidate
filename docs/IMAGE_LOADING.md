# Image loading: incremental step 03

`load_scene(scene_manifest, preprocess_config) -> ImageBatch` provides own
`ImageRecord` objects. It delegates the actual EXIF correction, RGB conversion,
resize/crop and normalization to the pinned DUSt3R loader. The original image.py
copyright and license remain in vendor; each batch records that source file/hash.
No vendor source is changed by this step.

The current policy is `dust3r_compatible_512_patch16_v1`: long_edge=512,
patch_size=16, and the explicit boolean square_ok. Policy expansion requires new
tests and schema support. JPG/JPEG/PNG are accepted; HEIC/HEIF are not supported.

Manifest order controls records. Optional image_id is honored; otherwise the file
path string is the stable image ID. IDs and canonical file paths must be unique.
Missing, unreadable, unsupported or hash-mismatched files fail. Identical bytes at
different paths generate an explicit warning in evidence instead of being hidden.

Sizes are H,W, pixels x,y, and crop boxes left,top,right,bottom with exclusive
right/bottom. Header EXIF metadata determines oriented size. Header inspection
does not resize or crop another image; actual pixel preprocessing happens once
through the pinned loader. Metadata includes raw/oriented/resize/processed size,
EXIF operation, exact crop bounds, original file hash, processed RGB hash and CPU
FP32 tensor hash. RGB is uint8 H,W,3; model tensors are FP32 1,3,H,W on CPU.

For integer pixel-center coordinates in the EXIF-oriented image:

```text
x_processed = sx * (x_oriented + 0.5) - 0.5 - crop_left
y_processed = sy * (y_oriented + 0.5) - 0.5 - crop_top
```

`A_oriented_to_processed` stores this transform using actual rounded resize
dimensions. It does not convert raw camera intrinsics through EXIF rotations or
mirrors, or provide depth outside the cropped image.

The incremental entry creates backend dictionaries at its adapter boundary from
own records. It passes original paths as identities and prepared tensors as image
data, preserving the pinned sparse-GA naming convention. Debug PNGs are never fed
back into the loader. Batch paths/order and manifest hash are checked against the
preflight evidence before model loading. Image metadata is written before model
execution, and the reconstruction report includes the processed-pixel fingerprint.

`scripts/check_image_loader.py` additionally loads reference tensors and executes
the recorded Pillow geometry to compare all pixels. This diagnostic deliberately
does extra CPU preprocessing; it does not run inference. CPU fixtures cover
landscape/portrait/square, all EXIF operations, RGBA/grayscale/JPEG conversion,
ordering/IDs, invalid inputs and marker positions under crop and resize.

```bash
python scripts/check_image_loader.py --config configs/scenes/scene_01.yaml
```

Evidence is written to work/image_loader_check/scene_01 by default. Actual
reconstruction still needs remote execution to validate the newly connected
loader; previous successful runs used the preceding entry version.
# Manifest pairing declarations

An optional `pairing` object may declare `order_semantics` (`capture_sequence` or
`unordered`) and `explicit_edges` (known image-ID pairs). Invalid/null declarations,
unknown fields, self loops and repeated/reverse edges are rejected before upstream
image loading. Missing metadata stays absent; filename order is not interpreted as
capture order. ImageBatch retains a copied declaration and writes it into
image_preprocessing.json when present. Pairing subsequently checks strategy-specific
requirements and connectivity. Existing manifests/preprocessing pixel hashes are
unchanged; adding declaration metadata changes the manifest file hash.
