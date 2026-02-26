import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from Model.model import TemporalSuperPointLite
from Evaluations.common import *

K_UZH_indoor = np.array([
    [278.66723066149086,   0.    , 319.75221200593535],
    [  0.    , 278.48991409740296, 241.96858910358173],
    [  0.    ,   0.   ,   1.   ]
], dtype=np.float64) # indoor

K_UZH_outdoor = np.array([
    [277.4786896484645,   0.    , 320.1052053576385],
    [  0.    , 277.42548548840034, 242.10083077857894],
    [  0.    ,   0.   ,   1.   ]
], dtype=np.float64) # outdoor



def quat_to_rot(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    n = np.sqrt(x*x + y*y + z*z + w*w)
    if n > 0:
        x /= n
        y /= n
        z /= n
        w /= n
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=np.float64)
    return R


def load_tum_poses_as_arrays(gt_path):
    ts_list = []
    T_list = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "" or line[0] == '#':
                continue
            tokens = line.split()
            if len(tokens) < 8:
                continue
            ts_str = tokens[0]
            tx, ty, tz = map(float, tokens[1:4])
            qx, qy, qz, qw = map(float, tokens[4:8])
            R = quat_to_rot(qx, qy, qz, qw)
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)

            ts_float = float(ts_str)
            ts_list.append(ts_float)
            T_list.append(T)

    ts_array = np.array(ts_list, dtype=np.float64)
    T_array = np.stack(T_list, axis=0) if len(T_list) > 0 else np.zeros((0, 4, 4))
    return ts_array, T_array


def associate_timestamps_tum_style(ts_img, ts_gt, max_diff=0.02):
    potential = []
    for i, t_i in enumerate(ts_img):
        for j, t_j in enumerate(ts_gt):
            diff = abs(t_i - t_j)
            if diff < max_diff:
                potential.append((diff, i, j))

    potential.sort(key=lambda x: x[0])

    matches = []
    used_i = set()
    used_j = set()
    for diff, i, j in potential:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        matches.append((i, j))

    matches.sort(key=lambda pair: ts_img[pair[0]])
    return matches


def skew_t(t):
    tx, ty, tz = t
    return np.array([
        [ 0,   -tz,  ty],
        [ tz,   0,  -tx],
        [-ty,  tx,   0 ]
    ], dtype=np.float64)


def fundamental_from_poses(T1, T2, K):
    T_12 = T2 @ np.linalg.inv(T1)  # camera1 -> camera2
    R = T_12[:3, :3]
    t = T_12[:3, 3]

    E = skew_t(t) @ R      # Essential matrix
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv  # Fundamental matrix
    return F


def load_uzh_left_images(meta_path, seq_root):
    ts_list = []
    paths = []
    with open(meta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "" or line[0] == '#':
                continue
            tokens = line.split()
            if len(tokens) < 3:
                continue
            ts = float(tokens[1])
            rel_path = "image_0/"+tokens[2].split("/")[1]
            img_path = os.path.join(seq_root, rel_path)
            ts_list.append(ts)
            paths.append(img_path)
    ts_img = np.array(ts_list, dtype=np.float64)
    return ts_img, paths

def compute_rep_mle_pair_F(kpts1, kpts2, F, shape, dist_thresh=3.0):

    def select_k_best(points, k):
        if points.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if points.shape[1] > 2:
            idx = points[:, 2].argsort()
            start = min(k, points.shape[0])
            pts_xy = points[idx][-start:, :2]
        else:
            start = min(k, points.shape[0])
            pts_xy = points[-start:, :2]
        return pts_xy.astype(np.float32)

    if kpts1 is None or kpts2 is None or kpts1.shape[0] == 0 or kpts2.shape[0] == 0:
        return 0.0, 0, -1.0

    pts1 = kpts1[:, :2].astype(np.float32)
    pts2 = kpts2[:, :2].astype(np.float32)
    N1, N2 = pts1.shape[0], pts2.shape[0]
    if N1 == 0 and N2 == 0:
        return 0.0, 0, -1.0

    x1_h = np.concatenate([pts1, np.ones((N1, 1), dtype=np.float32)], axis=1)  # [N1,3]
    x2_h = np.concatenate([pts2, np.ones((N2, 1), dtype=np.float32)], axis=1)  # [N2,3]

    l2 = (F @ x1_h.T).T   # [N1,3]
    denom2 = np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-8
    num2 = np.abs(x2_h @ l2.T).T           # [N1,N2]
    D12 = num2 / denom2[:, None]          # [N1,N2]

    l1 = (F.T @ x2_h.T).T                 # [N2,3]
    denom1 = np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-8
    num1 = np.abs(x1_h @ l1.T).T          # [N2,N1]
    D21 = num1 / denom1[:, None]          # [N2,N1]

    count1 = count2 = 0
    local_err1 = local_err2 = None

    if N2 > 0:
        min1 = D12.min(axis=1)
        local_err1 = min1[min1 <= dist_thresh]
        count1 = local_err1.size

    if N1 > 0:
        min2 = D21.min(axis=1)
        local_err2 = min2[min2 <= dist_thresh]
        count2 = local_err2.size

    rep = 0.0
    if N1 + N2 > 0:
        rep = (count1 + count2) / float(N1 + N2)

    num_matches = count1 + count2
    if num_matches > 0:
        err_sum = 0.0
        if local_err1 is not None and local_err1.size > 0:
            err_sum += local_err1.sum()
        if local_err2 is not None and local_err2.size > 0:
            err_sum += local_err2.sum()
        loc_err = err_sum / float(num_matches)
    else:
        loc_err = -1.0

    return float(rep), int(num_matches), float(loc_err)

def compute_rep_mle_uzh_sequence_with_F(
    seq_root,
    model,
    device="cuda",
    resize=(240, 320),
    dist_thresh=3.0,
    nms_dist=0,
    max_time_diff=0.02,
):

    meta_path = os.path.join(seq_root, "left_images.txt")
    gt_path   = os.path.join(seq_root, "groundtruth.txt")

    ts_img, img_paths = load_uzh_left_images(meta_path, seq_root)
    if ts_img.size == 0:
        print("No left_images in", seq_root)
        return 0.0, -1.0

    ts_gt, T_gt = load_tum_poses_as_arrays(gt_path)
    if ts_gt.size == 0:
        print("No GT poses in", gt_path)
        return 0.0, -1.0

    matches = associate_timestamps_tum_style(ts_img, ts_gt, max_diff=max_time_diff)
    if len(matches) < 2:
        print("Too few matches between images and GT in", seq_root)
        return 0.0, -1.0

    aligned_imgs = []
    aligned_T = []
    for idx_img, idx_gt in matches:
        aligned_imgs.append(img_paths[idx_img])
        aligned_T.append(T_gt[idx_gt])
    aligned_T = np.stack(aligned_T, axis=0)

    kpts_list = []
    pre_feat = None
    img_shape = None

    print(f"[{os.path.basename(seq_root)}] Aligned {len(aligned_imgs)} image-pose pairs, start extracting keypoints...")
    print("")

    if seq_root.split("/")[-1].split('_')[0] == "indoor":
        K_UZH = K_UZH_indoor
    else:
        K_UZH = K_UZH_outdoor

    for img_path in tqdm(aligned_imgs, desc=f"Extracting keypoints ({os.path.basename(seq_root)})"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            kpts_list.append(np.zeros((0, 3), dtype=np.float32))
            continue
        if img_shape is None:
            img_shape = img.shape

        kpts, _, pre_feat = extract_feature_descriptors_from_Ours(
            img, resize, model, pre_feat=pre_feat,
            device=device, nms_dist=nms_dist
        )
        kpts_list.append(kpts)

    reps = []
    total_matches = 0
    total_err_sum = 0.0

    for i in range(len(aligned_imgs) - 1):
        T1 = aligned_T[i]
        T2 = aligned_T[i + 1]
        F = fundamental_from_poses(T1, T2, K_UZH)

        kpts1 = kpts_list[i]
        kpts2 = kpts_list[i + 1]

        rep, num_matches, loc_err = compute_rep_mle_pair_F(
            kpts1, kpts2, F, img_shape,
            dist_thresh=dist_thresh
        )

        reps.append(rep)
        if num_matches > 0 and loc_err >= 0:
            total_matches += num_matches
            total_err_sum += loc_err * num_matches

    mean_rep = float(np.mean(reps)) if len(reps) > 0 else 0.0
    mle = total_err_sum / total_matches if total_matches > 0 else -1.0

    return mean_rep, mle

if __name__ == "__main__":
    device = "cuda"
    print("Using device:", device)

    base_path = "UZH FPV"   # 你自己改

    checkpoint_path = "Super_Changed_ORB.pth.tar"
    resize = (240, 320)

    model = TemporalSuperPointLite().to(device)
    model = load_model(model, checkpoint_path, device)
    model.eval()

    nms_dist = 8
    dist_thresh = 3.0
    max_time_diff = 0.02

    sequences = [
        'indoor_forward_3_snapdragon_with_gt',
        'indoor_forward_5_snapdragon_with_gt',
        'indoor_forward_6_snapdragon_with_gt',
        'indoor_forward_9_snapdragon_with_gt',
        'indoor_forward_10_snapdragon_with_gt',
        'outdoor_forward_1_snapdragon_with_gt',
        'outdoor_forward_3_snapdragon_with_gt',
    ]

    save_path = ""
    all_rep = []
    all_mle = []

    with open(save_path, "a+") as f:
        f.write(f"nms dist = {nms_dist}, dist_thresh = {dist_thresh}, max_time_diff = {max_time_diff}\n")

    for seq_name in sequences:
        seq_root = os.path.join(base_path, seq_name)
        print("\n=== UZH Sequence:", seq_name, "===")

        rep, mle = compute_rep_mle_uzh_sequence_with_F(
            seq_root,
            model,
            device=device,
            resize=resize,
            dist_thresh=dist_thresh,
            nms_dist=nms_dist,
            max_time_diff=max_time_diff,
        )

        all_rep.append(rep)
        all_mle.append(mle)

        print(f"{seq_name}  Rep = {rep:.4f},  MLE = {mle:.4f}")
        with open(save_path, "a+") as f:
            f.write(f"{seq_name}: Rep = {rep:.6f}, MLE = {mle:.6f}\n")

    mean_rep = float(np.mean(all_rep)) if len(all_rep) > 0 else 0.0
    mean_mle = float(np.mean(all_mle)) if len(all_mle) > 0 else -1.0

    print("\n========== UZH ==========")
    print("Rep: ", mean_rep)
    print("MLE: ", mean_mle)

