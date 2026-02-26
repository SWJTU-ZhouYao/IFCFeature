import os
import cv2, time
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns
from tqdm import tqdm

from Evaluations.common import *
from Model.model import TemporalSuperPointLite
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

def compute_MA_from_paths(path_list, ex_device="cpu", nms_dist=0):

    MA_List = []
    Num_List = []
    threshold_list = [1500, 2000, 2500, 3000]
    ref_img = cv2.imread(path_list[0], cv2.IMREAD_GRAYSCALE)
    pre_feat = None
    resize = (240, 320)
    device = torch.device(ex_device)
    model = TemporalSuperPointLite().to(device)
    checkpoint_path = "Super_Changed_ORB.pth.tar"
    model = load_model(model, checkpoint_path, device)
    model.eval()
    ref_kp, ref_des, pre_feat = extract_feature_descriptors_from_Ours(ref_img, resize, model, pre_feat=None, device=device, threshold=0.005, nms_dist=nms_dist)
    for index, path in enumerate(tqdm(path_list)):
        if index == 0:
            continue

        cur_img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        cur_kp, cur_des, pre_feat = extract_feature_descriptors_from_Ours(cur_img, resize, model, pre_feat, device=ex_device, threshold=0.005, nms_dist=nms_dist)
        matches = match_ORB(ref_des, cur_des, 0.75)
        if len(matches) >= 7:
            matches = sorted(matches, key=lambda x: x.distance)
        if len(matches) < 7:
            MA_List.append(0.5)
        else:
            _, accuracy_H, mask = compute_matching_accuracy(ref_kp, ref_des, cur_kp, cur_des,
                                                        true_matrix=None, method='F', ratio_test=0.75,
                                                        matches=matches)
            MA_List.append(accuracy_H)
        Num_List.append(len(matches))

        ref_kp = cur_kp
        ref_des = cur_des

    return MA_List, Num_List

if __name__ == '__main__':
    indices = ['indoor_forward_3_snapdragon_with_gt', 'indoor_forward_5_snapdragon_with_gt', 'indoor_forward_6_snapdragon_with_gt', 'indoor_forward_9_snapdragon_with_gt', 'indoor_forward_10_snapdragon_with_gt',
               'outdoor_forward_1_snapdragon_with_gt', 'outdoor_forward_3_snapdragon_with_gt']
    model_name = "UZH FPV"
    saved_model_name = "Ours_Super_Changed"
    nms_dist = 0
    max_time_diff = 0.02
    all_mean_MA = []
    all_mean_Num = []
    for sequence_name in indices:
        folder_path = "UZH FPV/"+sequence_name+"/image_0"
        seq_root = "UZH FPV/"+sequence_name
        file_names = os.listdir(folder_path)
        full_paths = sorted([os.path.join(folder_path, f) for f in file_names])
        MA_list, Num_list = compute_MA_from_paths(full_paths,ex_device="cuda",nms_dist=nms_dist)
        Average_MA = np.mean(MA_list)
        Average_Num = np.mean(Num_list)
        all_mean_MA.append(Average_MA)
        all_mean_Num.append(Average_Num)
        print(sequence_name, "AMA: ", Average_MA)
        print(sequence_name, " AMN: ", Average_Num)
