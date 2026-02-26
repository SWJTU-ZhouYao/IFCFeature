import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from OModel.model import TemporalSuperPointLite
from Evaluations.common import *

K_UZH_indoor = np.array([
    [278.66723066149086,   0.    , 319.75221200593535],
    [  0.    , 278.48991409740296, 241.96858910358173],
    [  0.    ,   0.   ,   1.   ]
], dtype=np.float64)

K_UZH_outdoor = np.array([
    [277.4786896484645,   0.    , 320.1052053576385],
    [  0.    , 277.42548548840034, 242.10083077857894],
    [  0.    ,   0.   ,   1.   ]
], dtype=np.float64)


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

    E = skew_t(t) @ R
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ E @ Kinv
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
            rel_path = "image_0/" + tokens[2].split("/")[1]
            img_path = os.path.join(seq_root, rel_path)
            ts_list.append(ts)
            paths.append(img_path)
    ts_img = np.array(ts_list, dtype=np.float64)
    return ts_img, paths

def compute_nn_labels_scores_pair_F(kpts1, desc1, kpts2, desc2, F, epi_thresh=3.0):
    if kpts1 is None or kpts2 is None:
        return [], []
    if kpts1.shape[0] == 0 or kpts2.shape[0] == 0:
        return [], []
    if desc1 is None or desc2 is None:
        return [], []

    kpts1 = np.asarray(kpts1, dtype=np.float32)
    kpts2 = np.asarray(kpts2, dtype=np.float32)
    desc1 = np.asarray(desc1, dtype=np.float32)
    desc2 = np.asarray(desc2, dtype=np.float32)

    N1, N2 = kpts1.shape[0], kpts2.shape[0]
    if N1 == 0 or N2 == 0:
        return [], []

    pts1 = kpts1[:, :2]
    pts2 = kpts2[:, :2]

    x1_h = np.concatenate([pts1, np.ones((N1, 1), dtype=np.float32)], axis=1)  # [N1,3]
    x2_h = np.concatenate([pts2, np.ones((N2, 1), dtype=np.float32)], axis=1)  # [N2,3]

    l2 = (F @ x1_h.T).T                  # [N1,3]
    denom2 = np.sqrt(l2[:, 0]**2 + l2[:, 1]**2) + 1e-8
    num2 = np.abs(x2_h @ l2.T).T         # [N1,N2]
    D12 = num2 / denom2[:, None]        # [N1,N2]

    l1 = (F.T @ x2_h.T).T               # [N2,3]
    denom1 = np.sqrt(l1[:, 0]**2 + l1[:, 1]**2) + 1e-8
    num1 = np.abs(x1_h @ l1.T).T        # [N2,N1]
    D21 = num1 / denom1[:, None]        # [N2,N1]

    desc1_norm = desc1 / (np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-8)
    desc2_norm = desc2 / (np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-8)

    sim = desc1_norm @ desc2_norm.T     # [N1,N2]

    scores = []
    labels = []

    for i in range(N1):
        j_nn = np.argmax(sim[i])
        s_ij = sim[i, j_nn]

        d12 = D12[i, j_nn]
        d21 = D21[j_nn, i]
        e_sym = 0.5 * (d12 + d21)

        label = 1 if e_sym <= epi_thresh else 0

        scores.append(float(s_ij))
        labels.append(int(label))

    return scores, labels

def compute_nn_map_uzh_sequence_with_F(
    seq_root,
    model,
    device="cuda",
    resize=(240, 320),
    epi_thresh=3.0,
    nms_dist=0,
    max_time_diff=0.02,
):

    meta_path = os.path.join(seq_root, "left_images.txt")
    gt_path   = os.path.join(seq_root, "groundtruth.txt")

    ts_img, img_paths = load_uzh_left_images(meta_path, seq_root)
    if ts_img.size == 0:
        print("No left_images in", seq_root)
        return 0.0, [], []

    ts_gt, T_gt = load_tum_poses_as_arrays(gt_path)
    if ts_gt.size == 0:
        print("No GT poses in", gt_path)
        return 0.0, [], []

    matches = associate_timestamps_tum_style(ts_img, ts_gt, max_diff=max_time_diff)
    if len(matches) < 2:
        print("Too few matches between images and GT in", seq_root)
        return 0.0, [], []

    aligned_imgs = []
    aligned_T = []
    for idx_img, idx_gt in matches:
        aligned_imgs.append(img_paths[idx_img])
        aligned_T.append(T_gt[idx_gt])
    aligned_T = np.stack(aligned_T, axis=0)

    if seq_root.split("/")[-1].split('_')[0] == "indoor":
        K_UZH = K_UZH_indoor
    else:
        K_UZH = K_UZH_outdoor

    kpts_list = []
    desc_list = []
    pre_feat = None
    img_shape = None

    print(f"[{os.path.basename(seq_root)}] Aligned {len(aligned_imgs)} image-pose pairs, start extracting keypoints+descriptors...")
    for img_path in tqdm(aligned_imgs, desc=f"Extracting ({os.path.basename(seq_root)})"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            kpts_list.append(np.zeros((0, 3), dtype=np.float32))
            desc_list.append(np.zeros((0, 256), dtype=np.float32))
            continue
        if img_shape is None:
            img_shape = img.shape

        kpts, desc, pre_feat = extract_feature_descriptors_from_Ours(
            img, resize, model, pre_feat=pre_feat,
            device=device, nms_dist=nms_dist
        )
        kpts_list.append(kpts)
        desc_list.append(desc)

    all_scores = []
    all_labels = []

    for i in range(len(aligned_imgs) - 1):
        T1 = aligned_T[i]
        T2 = aligned_T[i + 1]
        F = fundamental_from_poses(T1, T2, K_UZH)

        kpts1 = kpts_list[i]
        kpts2 = kpts_list[i + 1]
        desc1 = desc_list[i]
        desc2 = desc_list[i + 1]

        scores_pair, labels_pair = compute_nn_labels_scores_pair_F(
            kpts1, desc1, kpts2, desc2, F, epi_thresh=epi_thresh
        )

        all_scores.extend(scores_pair)
        all_labels.extend(labels_pair)

    if len(all_labels) == 0 or np.sum(all_labels) == 0:
        ap_seq = 0.0
    else:
        all_scores_arr = np.array(all_scores, dtype=np.float32)
        all_labels_arr = np.array(all_labels, dtype=np.int32)
        ap_seq = float(average_precision_score(all_labels_arr, all_scores_arr))

    return ap_seq, all_scores, all_labels

if __name__ == "__main__":
    device = "cuda"
    print("Using device:", device)

    base_path = "UZH FPV"   # 你自己的 UZH 根目录
    checkpoint_path = "Super_Changed_ORB.pth.tar"
    resize = (240, 320)

    model = TemporalSuperPointLite().to(device)
    model = load_model(model, checkpoint_path, device)
    model.eval()

    nms_dist = 0
    epi_thresh = 3.0
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

    all_seq_ap = []
    global_scores = []
    global_labels = []

    with open(save_path, "a+") as f:
        f.write(f"nms_dist={nms_dist}, epi_thresh={epi_thresh}, max_time_diff={max_time_diff}\n")

    for seq_name in sequences:
        seq_root = os.path.join(base_path, seq_name)
        print(f"\n===  UZH Sequence: {seq_name} ===")

        ap_seq, scores_seq, labels_seq = compute_nn_map_uzh_sequence_with_F(
            seq_root,
            model,
            device=device,
            resize=resize,
            epi_thresh=epi_thresh,
            nms_dist=nms_dist,
            max_time_diff=max_time_diff,
        )

        all_seq_ap.append(ap_seq)
        global_scores.extend(scores_seq)
        global_labels.extend(labels_seq)

        print(f"{seq_name}  NN mAP = {ap_seq:.4f}")
        with open(save_path, "a+") as f:
            f.write(f"{seq_name}: AP = {ap_seq:.6f}\n")

    if len(global_labels) > 0 and np.sum(global_labels) > 0:
        global_ap = float(average_precision_score(
            np.array(global_labels, dtype=np.int32),
            np.array(global_scores, dtype=np.float32)
        ))
    else:
        global_ap = 0.0

    mean_seq_ap = float(np.mean(all_seq_ap)) if len(all_seq_ap) > 0 else 0.0

    print("\n========== UZH NN mAP ==========")
    print("NN mAP   :", global_ap)

