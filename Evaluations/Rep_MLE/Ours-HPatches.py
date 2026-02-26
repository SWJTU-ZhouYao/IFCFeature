import os
import cv2
import numpy as np
import torch
from Model.model import TemporalSuperPointLite
from Evaluations.common import *

def warp_keypoints(kpts, H):
    if kpts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    pts = kpts[:, :2]  # [N,2]
    pts_h = np.concatenate(
        [pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1
    )  # [N,3]
    pts_w = pts_h @ H.T  # [N,3]
    pts_w = pts_w[:, :2] / pts_w[:, 2:3]
    return pts_w.astype(np.float32)

def keep_points_in_image(pts, img_shape):
    H, W = img_shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    return mask

def compute_repeatability_pair(kpts1, kpts2, H_1_2, shape1, shape2, dist_thresh=3.0):
    if kpts1.shape[0] == 0 or kpts2.shape[0] == 0:
        return 0.0, 0, 0, 0, -1.0
    pts1_w = warp_keypoints(kpts1, H_1_2)  # [N1,2]
    mask1 = keep_points_in_image(pts1_w, shape2)
    pts1_w = pts1_w[mask1]
    pts1 = kpts1[mask1, :2]
    N1_overlap = pts1.shape[0]
    H_2_1 = np.linalg.inv(H_1_2)
    pts2_w = warp_keypoints(kpts2, H_2_1)
    mask2 = keep_points_in_image(pts2_w, shape1)
    pts2 = kpts2[mask2, :2]
    N2_overlap = pts2.shape[0]

    if N1_overlap == 0 or N2_overlap == 0:
        return 0.0, 0, N1_overlap, N2_overlap, -1.0
    diff = pts1_w[:, None, :] - pts2[None, :, :]  # [N1,N2,2]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))    # [N1,N2]
    nn_j = np.argmin(dists, axis=1)              # [N1]
    nn_dist = dists[np.arange(dists.shape[0]), nn_j]
    cand_idx = np.where(nn_dist <= dist_thresh)[0]
    if cand_idx.size == 0:
        return 0.0, 0, N1_overlap, N2_overlap, -1.0
    candidates = [(int(i), int(nn_j[i]), float(nn_dist[i])) for i in cand_idx]
    candidates.sort(key=lambda x: x[2])

    used1 = set()
    used2 = set()
    num_matches = 0
    sum_dist = 0.0
    for i, j, d in candidates:
        if i in used1 or j in used2:
            continue
        used1.add(i)
        used2.add(j)
        num_matches += 1
        sum_dist += d

    if num_matches == 0:
        return 0.0, 0, N1_overlap, N2_overlap, -1.0

    rep = num_matches / float(min(N1_overlap, N2_overlap))
    loc_err = sum_dist / float(num_matches)

    return rep, num_matches, N1_overlap, N2_overlap, loc_err

def load_hpatches_gray(seq_dir, idx):
    filename_candidates = [f"{idx}.ppm", f"{idx}.png", f"{idx}.jpg"]
    for name in filename_candidates:
        path = os.path.join(seq_dir, name)
        if os.path.exists(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            return img, path
    raise FileNotFoundError(f"Cannot find image {idx} in {seq_dir}")

def load_hpatches_homography(seq_dir, idx):
    base = f"H_1_{idx}"
    path_candidates = [os.path.join(seq_dir, base),
                       os.path.join(seq_dir, base + ".txt")]
    for p in path_candidates:
        if os.path.exists(p):
            H = np.loadtxt(p)
            return H, p
    raise FileNotFoundError(f"Cannot find homography {base} in {seq_dir}")

def evaluate_hpatches_repeatability(model, hpatches_root,
                                    device="cpu", resize=(240, 320),
                                    dist_thresh=3.0, nms_dist=0):
    all_scores = []
    illum_scores = []     # i_*
    viewpoint_scores = [] # v_*

    all_les = []
    illum_les = []
    viewpoint_les = []

    seq_names = sorted(os.listdir(hpatches_root))
    print(f"Found {len(seq_names)} sequences under {hpatches_root}")

    for seq_idx, seq_name in enumerate(seq_names):
        seq_dir = os.path.join(hpatches_root, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        if not (seq_name.startswith("i_") or seq_name.startswith("v_")):
            continue

        try:
            img1, img1_path = load_hpatches_gray(seq_dir, 1)
        except FileNotFoundError:
            continue

        H1, W1 = img1.shape

        kpts1, _, pre_feat = extract_feature_descriptors_from_Ours(
            img1, resize, model, pre_feat=None, device=device, nms_dist=nms_dist
        )

        seq_pair_scores = []
        seq_pair_les = []

        for idx_img in range(2, 7):
            try:
                img2, img2_path = load_hpatches_gray(seq_dir, idx_img)
                H_1_2, H_path = load_hpatches_homography(seq_dir, idx_img)
            except FileNotFoundError:
                continue

            H2, W2 = img2.shape

            kpts2, _, _ = extract_feature_descriptors_from_Ours(
                img2, resize, model, pre_feat=pre_feat, device=device, nms_dist=nms_dist
            )

            rep, num_matches, N1, N2, loc_err = compute_repeatability_pair(
                kpts1, kpts2, H_1_2, (H1, W1), (H2, W2), dist_thresh=dist_thresh
            )

            all_scores.append(rep)
            seq_pair_scores.append(rep)
            if seq_name.startswith("i_"):
                illum_scores.append(rep)
            else:
                viewpoint_scores.append(rep)

            if loc_err >= 0:
                all_les.append(loc_err)
                seq_pair_les.append(loc_err)
                if seq_name.startswith("i_"):
                    illum_les.append(loc_err)
                else:
                    viewpoint_les.append(loc_err)

        if len(seq_pair_scores) > 0:
            mean_seq = float(np.mean(seq_pair_scores))
            mean_seq_le = float(np.mean(seq_pair_les)) if len(seq_pair_les) > 0 else float("nan")
            print(f"[{seq_idx:03d}] {seq_name}: mean repeat = {mean_seq:.4f}, "
                  f"mean LE = {mean_seq_le:.4f} over {len(seq_pair_scores)} pairs")

    def safe_mean(x):
        return float(np.mean(x)) if len(x) > 0 else 0.0

    print("\n========== HPatches Repeatability (epsilon = %.1f) ==========" % dist_thresh)
    print(f"Illumination(i): {len(illum_scores):4d}, mean repeat = {safe_mean(illum_scores):.4f}")

    print("\n========== HPatches Localization Error (MLE) ==========")
    print(f"Illumination(i): {len(illum_les):4d}, mean LE = {safe_mean(illum_les):.4f}")

    return {
        "illum_pairs": illum_scores,
        "illum_les": illum_les,
    }

if __name__ == "__main__":
    device = "cuda"
    print("Using device:", device)

    checkpoint_path = "Super_Changed_ORB.pth.tar"
    hpatches_root = "hpatches-sequences-release"
    resize = (240, 320)
    model = TemporalSuperPointLite().to(device)
    model = load_model(model, checkpoint_path, device)
    model.eval()
    results = evaluate_hpatches_repeatability(
        model,
        hpatches_root=hpatches_root,
        device=device,
        resize=resize,
        dist_thresh=3.0,
        nms_dist=0
    )

