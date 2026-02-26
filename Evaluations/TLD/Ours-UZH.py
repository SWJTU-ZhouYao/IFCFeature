import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter

from Model.model import TemporalSuperPointLite
from Evaluations.common import *


class MapPoint:
    __slots__ = ("id", "obs")

    def __init__(self, mp_id: int):
        self.id = mp_id
        self.obs = 1


if __name__ == "__main__":
    device = "cuda"
    print("Using device:", device)

    base_path = "UZH FPV"   #
    checkpoint_path = "Super_Changed_ORB.pth.tar"
    resize = (240, 320)

    model = TemporalSuperPointLite().to(device)
    model = load_model(model, checkpoint_path, device)
    model.eval()

    nms_dist = 0
    ratio_th = 0.75
    det_thresh = 0.005
    indices = [
        'indoor_forward_3_snapdragon_with_gt',
        'indoor_forward_5_snapdragon_with_gt',
        'indoor_forward_6_snapdragon_with_gt',
        'indoor_forward_9_snapdragon_with_gt',
        'indoor_forward_10_snapdragon_with_gt',
        'outdoor_forward_1_snapdragon_with_gt',
        'outdoor_forward_3_snapdragon_with_gt',
    ]

    global_counter = Counter()
    per_seq_dists = {}

    save_path = ""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for sequence_name in indices:
        print("\n==============================")
        folder_path = os.path.join(base_path, sequence_name, "image_0")
        file_names = sorted(os.listdir(folder_path))
        full_paths = sorted([os.path.join(folder_path, f) for f in file_names])
        first_img = cv2.imread(full_paths[0], cv2.IMREAD_GRAYSCALE)
        ref_kps, ref_desc, ref_pre_feat = extract_feature_descriptors_from_Ours(
            first_img, resize, model,
            pre_feat=None,
            device=device,
            nms_dist=nms_dist,
            threshold=det_thresh
        )

        map_points = []
        prev_map_ids = np.empty(ref_kps.shape[0], dtype=int)

        for i in range(ref_kps.shape[0]):
            mp_id = len(map_points)
            mp = MapPoint(mp_id)
            map_points.append(mp)
            prev_map_ids[i] = mp_id

        for index, file_path in enumerate(tqdm(full_paths, desc=f"{sequence_name}")):
            if index == 0:
                continue

            cur_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

            cur_kps, cur_desc, cur_pre_feat = extract_feature_descriptors_from_Ours(
                cur_img, resize, model,
                pre_feat=ref_pre_feat,
                device=device,
                nms_dist=nms_dist,
                threshold=det_thresh
            )

            if cur_kps is None or cur_kps.shape[0] == 0 or cur_desc is None or cur_desc.shape[0] == 0:
                ref_kps = cur_kps
                ref_desc = cur_desc
                ref_pre_feat = cur_pre_feat
                prev_map_ids = np.empty(0, dtype=int)
                continue

            if ref_desc is None or ref_desc.shape[0] == 0:
                matches = []
            else:
                raw_matches = match_ORB(ref_desc, cur_desc, ratio_th)
                flat_matches = []
                for item in raw_matches:
                    if hasattr(item, "distance"):
                        flat_matches.append(item)
                    elif isinstance(item, (list, tuple)):
                        for m in item:
                            if hasattr(m, "distance"):
                                flat_matches.append(m)
                matches = sorted(flat_matches, key=lambda x: x.distance)

            num_cur = cur_kps.shape[0]
            cur_map_ids = np.full(num_cur, -1, dtype=int)

            if len(matches) != 0:
                for m in matches:
                    q_idx = m.queryIdx
                    t_idx = m.trainIdx

                    if q_idx < 0 or q_idx >= prev_map_ids.shape[0]:
                        continue
                    if t_idx < 0 or t_idx >= num_cur:
                        continue

                    mp_id = prev_map_ids[q_idx]
                    if mp_id < 0 or mp_id >= len(map_points):
                        continue

                    if cur_map_ids[t_idx] != -1:
                        continue

                    cur_map_ids[t_idx] = mp_id
                    map_points[mp_id].obs += 1

            for k_idx in range(num_cur):
                if cur_map_ids[k_idx] == -1:
                    mp_id = len(map_points)
                    mp = MapPoint(mp_id)
                    map_points.append(mp)
                    cur_map_ids[k_idx] = mp_id

            ref_kps = cur_kps
            ref_desc = cur_desc
            ref_pre_feat = cur_pre_feat
            prev_map_ids = cur_map_ids

        track_lengths = [mp.obs for mp in map_points]
        seq_counter = Counter(track_lengths)
        per_seq_dists[sequence_name] = {str(k): int(v) for k, v in sorted(seq_counter.items())}

        global_counter.update(track_lengths)

        print(f" {sequence_name} map number: {len(map_points)}")
        print(f"  Track Length Distribution: {sorted(seq_counter.items())[:10]}")
    global_dist = {str(k): int(v) for k, v in sorted(global_counter.items())}

    with open(save_path, "w") as f:
        f.write("Per-sequence Track Length Distributions:\n")
        for seq_name in indices:
            if seq_name in per_seq_dists:
                f.write(seq_name + ":\n")
                f.write(str(per_seq_dists[seq_name]) + "\n")
        f.write("\nGlobal Track Length Distribution:\n")
        f.write(str(global_dist) + "\n")
