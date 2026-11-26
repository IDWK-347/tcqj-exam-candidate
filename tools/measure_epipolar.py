#!/usr/bin/env python3
"""Measure epipolar consistency of exported cameras against fixed SIFT matches.

Conventions (verified from artifacts/cameras.json):
  T_world_from_camera maps camera coordinates to the standard world:
      X_world = R @ X_cam + t
  Therefore the relative pose from camera 1 to camera 2 is
      X_cam2 = R2^T R1 X_cam1 + R2^T (t1 - t2)
      R_rel = R2^T R1,  t_rel = R2^T (t1 - t2)
  and the pixel fundamental matrix is
      F = K2^-T [t_rel]_x R_rel K1^-1

Statistic: Sampson distance in processed pixels (sqrt of the squared
Sampson distance), evaluated on one fixed SIFT+ratio-match set per pair and
one fixed RANSAC inlier set per pair (seed fixed), so different runs are
compared on exactly the same correspondences.

The optional --legacy mode reproduces the superseded DIAGNOSIS.md numbers:
it uses the opposite relative-pose direction and treats E as F without
K-normalization. It exists only to explain the old values, not as a metric.

By default keypoints are mapped to processed coordinates by pure scaling
(512/w, 384/h). Pass --preprocess-affine to use each image's exact
A_oriented_to_processed matrix from image_preprocessing.json instead, which
includes the sub-pixel translation; that is the faithful coordinate mapping.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

RATIO = 0.75
RANSAC_PX = 1.5
RANSAC_CONF = 0.999
SEED = 0
PROCESSED_HW = (384, 512)  # H, W


def _skew(t):
    return np.array([[0.0, -t[2], t[1]], [t[2], 0.0, -t[0]], [-t[1], t[0], 0.0]])


def _load_cameras(run_dir):
    path = Path(run_dir) / "artifacts" / "cameras.json"
    raw = json.loads(path.read_text())
    cameras = raw["cameras"]
    out = {}
    for cam in cameras:
        hw = tuple(cam["processed_hw"])
        if hw != PROCESSED_HW:
            raise SystemExit(f"{cam['image_id']}: processed_hw {hw} != {PROCESSED_HW}")
        K = np.asarray(cam["K_processed"], dtype=np.float64)
        if K.shape != (3, 3):
            raise SystemExit(f"{cam['image_id']}: K shape {K.shape}")
        if not np.allclose([K[0, 2], K[1, 2]], [PROCESSED_HW[1] / 2, PROCESSED_HW[0] / 2]):
            raise SystemExit(f"{cam['image_id']}: principal point is not processed-image center")
        T_wc = np.asarray(cam["T_world_from_camera"], dtype=np.float64)
        T_cw = np.asarray(cam["T_camera_from_world"], dtype=np.float64)
        if not np.allclose(T_wc @ T_cw, np.eye(4), atol=1e-8):
            raise SystemExit(f"{cam['image_id']}: T_world_from_camera @ T_camera_from_world != I")
        if not np.allclose(T_wc[3], [0, 0, 0, 1], atol=1e-9):
            raise SystemExit(f"{cam['image_id']}: malformed T_world_from_camera")
        out[cam["image_id"]] = {"K": K, "T_wc": T_wc, "T_cw": T_cw, "hw": hw}
    return out


def _load_affines(run_dir):
    """Per-image oriented->processed affine matrices, from image_preprocessing.json."""
    raw = json.loads((Path(run_dir) / "image_preprocessing.json").read_text())
    return {rec["image_id"]: np.asarray(rec["A_oriented_to_processed"], dtype=np.float64)
            for rec in raw["records"]}


def _detect(scene_dir, image_ids, affines=None):
    sift = cv2.SIFT_create()
    feats = {}
    for name in image_ids:
        path = Path(scene_dir) / name
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"cannot read {path}")
        h, w = image.shape
        keypoints, desc = sift.detectAndCompute(image, None)
        pts = np.asarray([kp.pt for kp in keypoints], dtype=np.float64)
        if affines is None:
            # Approximation: pure scale, omits the sub-pixel translation.
            pts[:, 0] *= PROCESSED_HW[1] / w
            pts[:, 1] *= PROCESSED_HW[0] / h
        else:
            A = affines[name]
            homo = np.c_[pts, np.ones(len(pts))]
            pts = (A @ homo.T).T[:, :2]
        feats[name] = (pts, desc)
    return feats


def _fixed_matches(feats, a, b):
    pa, da = feats[a]
    pb, db = feats[b]
    bf = cv2.BFMatcher()
    pairs = bf.knnMatch(da, db, k=2)
    good = [m for m, n in pairs if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return None
    p1 = np.asarray([pa[m.queryIdx] for m in good], dtype=np.float64)
    p2 = np.asarray([pb[m.trainIdx] for m in good], dtype=np.float64)
    # Fix the RANSAC sample sequence so every run uses the same inliers.
    cv2.setRNGSeed(SEED)
    F, mask = cv2.findFundamentalMat(p1, p2, cv2.FM_RANSAC, RANSAC_PX, RANSAC_CONF)
    if F is None or F.shape != (3, 3) or mask is None:
        return None
    inliers = mask.ravel().astype(bool)
    return {"p1": p1[inliers], "p2": p2[inliers], "matches": len(good),
            "inliers": int(inliers.sum()), "F_ransac": F}


def _relative(cam1, cam2, legacy=False):
    T1, T2 = cam1["T_wc"], cam2["T_wc"]
    R1, t1 = T1[:3, :3], T1[:3, 3]
    R2, t2 = T2[:3, :3], T2[:3, 3]
    C1 = -R1.T @ t1  # camera center in world
    C2 = -R2.T @ t2
    if legacy:
        # Superseded DIAGNOSIS.md direction: camera2 -> camera1, using centers.
        R = R1.T @ R2
        t = R1.T @ (C2 - C1)
    else:
        # X_cam2 = R2^T R1 X_cam1 + R2^T (t1 - t2)
        R = R2.T @ R1
        t = R2.T @ (t1 - t2)
    norm = np.linalg.norm(t)
    if not np.isfinite(norm) or norm == 0:
        raise SystemExit("degenerate relative translation")
    return _skew(t / norm) @ R


def _fundamental(E, K1, K2, legacy=False):
    if legacy:
        return E  # E applied directly to pixel coordinates: wrong units
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def _sampson(F, x1, x2):
    h1 = np.c_[x1, np.ones(len(x1))]
    h2 = np.c_[x2, np.ones(len(x2))]
    Fx1 = (F @ h1.T).T
    Ftx2 = (F.T @ h2.T).T
    numerator = np.sum(h2 * Fx1, axis=1) ** 2
    denominator = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def _parse_pairs(text, cameras):
    if text:
        pairs = []
        for item in text.split(","):
            a, b = item.strip().split("-")
            pairs.append((a if a.endswith(".png") else a + ".png",
                          b if b.endswith(".png") else b + ".png"))
    else:
        ids = list(cameras)
        pairs = [("01.png", "02.png"), ("04.png", "05.png"), ("01.png", "07.png")]
        pairs = [p for p in pairs if p[0] in cameras and p[1] in cameras]
        if not pairs and len(ids) == 2:
            pairs = [(ids[0], ids[1])]
    for a, b in pairs:
        if a not in cameras or b not in cameras:
            raise SystemExit(f"pair {a}-{b} not in cameras.json")
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scene", required=True, help="scene name, e.g. scene_b")
    parser.add_argument("--run", action="append", required=True, help="run directory")
    parser.add_argument("--pairs", help="comma list like 01-02,04-05,01-07")
    parser.add_argument("--legacy", action="store_true", help="reproduce superseded metric")
    parser.add_argument("--preprocess-affine", action="store_true",
                        help="use image_preprocessing.json's affine (includes the "
                             "sub-pixel translation) instead of pure scaling")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    scene_dir = Path("data") / args.scene
    if not scene_dir.is_dir():
        raise SystemExit(f"missing scene directory: {scene_dir}")

    prepared = []
    for run_dir in args.run:
        cameras = _load_cameras(run_dir)
        pairs = _parse_pairs(args.pairs, cameras)
        prepared.append((run_dir, cameras, pairs))

    image_ids = sorted({i for _, _, pairs in prepared for pair in pairs for i in pair})

    affines = None
    if args.preprocess_affine:
        affines = _load_affines(args.run[0])
        for run_dir in args.run[1:]:
            other = _load_affines(run_dir)
            for name in image_ids:
                if name not in other or not np.allclose(other[name], affines[name]):
                    raise SystemExit(f"{run_dir}: affine for {name} differs from {args.run[0]}")
        missing = [i for i in image_ids if i not in affines]
        if missing:
            raise SystemExit(f"{args.run[0]}: no affine for {missing}")

    feats = _detect(scene_dir, image_ids, affines)

    cache = {}
    for _, cameras, pairs in prepared:
        for a, b in pairs:
            if (a, b) not in cache:
                cache[(a, b)] = _fixed_matches(feats, a, b)

    rows = []
    for run_dir, cameras, pairs in prepared:
        for a, b in pairs:
            fixed = cache[(a, b)]
            if fixed is None:
                rows.append({"run": run_dir, "pair": f"{a}-{b}", "status": "no_matches"})
                continue
            E = _relative(cameras[a], cameras[b], legacy=args.legacy)
            F = _fundamental(E, cameras[a]["K"], cameras[b]["K"], legacy=args.legacy)
            err = _sampson(F, fixed["p1"], fixed["p2"])
            control = _sampson(fixed["F_ransac"], fixed["p1"], fixed["p2"])
            rows.append({
                "run": run_dir, "pair": f"{a}-{b}", "mode": "legacy" if args.legacy else "correct",
                "matches": fixed["matches"], "inliers": fixed["inliers"],
                "median_px": float(np.median(err)), "mean_px": float(np.mean(err)),
                "p90_px": float(np.percentile(err, 90)),
                "control_median_px": float(np.median(control)),
            })

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    header = ("run,pair,mode,matches,inliers,median_px,mean_px,p90_px,"
              "control_median_px")
    print(header)
    for row in rows:
        if row.get("status") == "no_matches":
            print(f"{row['run']},{row['pair']},no_matches")
            continue
        print(f"{row['run']},{row['pair']},{row['mode']},{row['matches']},"
              f"{row['inliers']},{row['median_px']:.3f},{row['mean_px']:.3f},"
              f"{row['p90_px']:.3f},{row['control_median_px']:.3f}")


if __name__ == "__main__":
    main()