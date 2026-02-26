import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from Model.model import TemporalSuperPointLite
from Evaluations.common import *

if __name__ == "__main__":
    device = "cuda"
    print("Using device:", device)

    base_path = "UZH FPV"
    checkpoint_path = "Super_Changed_ORB.pth.tar"
    resize = (240, 320)

    model = TemporalSuperPointLite().to(device)
    model = load_model(model, checkpoint_path, device)
    model.eval()

    nms_dist = 0
    epi_thresh = 3.0
    max_time_diff = 0.02

    indices = ['indoor_forward_3_snapdragon_with_gt', 'indoor_forward_5_snapdragon_with_gt', 'indoor_forward_6_snapdragon_with_gt', 'indoor_forward_9_snapdragon_with_gt', 'indoor_forward_10_snapdragon_with_gt',
               'outdoor_forward_1_snapdragon_with_gt', 'outdoor_forward_3_snapdragon_with_gt']
    sr = []
    for sequence_name in indices:
        folder_path = "UZH FPV/" + sequence_name + "/image_0"
        file_names = os.listdir(folder_path)
        full_paths = sorted([os.path.join(folder_path, f) for f in file_names])
        img_num = len(full_paths)
        srs = []
        for index, file_path in enumerate(tqdm(full_paths)):
            if index+10 >= img_num:
                continue
            ref_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            next_img = cv2.imread(full_paths[index+1], cv2.IMREAD_GRAYSCALE)
            last_img = cv2.imread(full_paths[index+10], cv2.IMREAD_GRAYSCALE)
            ref_kp, ref_desc, ref_pre_feat = extract_feature_descriptors_from_Ours(ref_img, resize, model, pre_feat=None, device=device, nms_dist=0, threshold=0.005)
            next_kp, next_desc, next_pre_feat = extract_feature_descriptors_from_Ours(next_img, resize, model, pre_feat=ref_pre_feat, device=device, nms_dist=0, threshold=0.005)
            last_kp, last_desc, last_pre_feat = extract_feature_descriptors_from_Ours(last_img, resize, model, pre_feat=ref_pre_feat, device=device, nms_dist=0, threshold=0.005)

            next_matches = match_ORB(ref_desc, next_desc, 0.75)
            if len(next_matches) >= 7:
                next_matches = sorted(next_matches, key=lambda x: x.distance)
            else:
                continue

            last_matches = match_ORB(ref_desc, last_desc, 0.75)
            if len(last_matches) >= 7:
                last_matches = sorted(last_matches, key=lambda x: x.distance)
            else:
                continue

            srs.append(float(len(last_matches)) / float(len(next_matches)))
        print(f"{sequence_name}: {np.mean(srs)}")
        sr.append(np.mean(srs))
    print(f"AKSR: {np.mean(sr)}")
