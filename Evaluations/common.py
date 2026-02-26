import torch
import cv2
import numpy as np
import scipy.spatial as spatial
import torch.nn.functional as F
from skimage.exposure import match_histograms
from torchvision import transforms
import matplotlib.pyplot as plt
def to_cv2_keypoints(kpts):
    return [cv2.KeyPoint(float(x), float(y), 1) for (x, y, z) in kpts]

def load_model(model, checkpoint_path, device=torch.device("cpu")):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    return model

def match_ORB(des1, des2, ratio_test=0):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    if ratio_test:
        raw_matches = bf.knnMatch(des1, des2, k=2)
        matches = []
        for m, n in raw_matches:
            if m.distance < ratio_test * n.distance:
                matches.append(m)
    else:
        matches = bf.match(des1, des2)
    if len(matches) == 0:
        return [], 0.0, None
    return matches

def extract_feature_descriptors_from_Ours(img, resize, model, pre_feat=None, device="cpu", nms_dist=2, threshold=0.005):
    H0, W0 = img.shape[:2]
    if resize:
        img_resized = cv2.resize(img, (resize[1], resize[0]))
    else:
        img_resized = img
    tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor, prev_feat=pre_feat)
        semi = output["detector_logits"]  # [1,65,H/8,W/8]
        desc = output["descriptor_map"]  # [1,256,H/8,W/8]
        pre_feat = output["prev_feat"]
    kpts_resized = get_keypoints_from_semi_Ours(semi, threshold=threshold, nms_dist=nms_dist)
    descriptors = extract_descriptors(desc, kpts_resized)
    if resize:
        scale_x = W0 / resize[1] #(240,320)
        scale_y = H0 / resize[0]
        kpts = np.zeros_like(kpts_resized)
        kpts[:, 0] = kpts_resized[:, 0] * scale_x  # x
        kpts[:, 1] = kpts_resized[:, 1] * scale_y  # y
        kpts[:, 2] = kpts_resized[:, 2]            # score
    else:
        kpts = kpts_resized

    return kpts, descriptors, pre_feat
def extract_feature_descriptors_no_nms_from_Ours(img, resize, model, pre_feat=None, device="cpu"):
    H0, W0 = img.shape[:2]
    if resize:
        img_resized = cv2.resize(img, (resize[1], resize[0]))
    else:
        img_resized = img
    tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor, prev_feat=pre_feat)
        semi = output["detector_logits"]  # [1,65,H/8,W/8]
        desc = output["descriptor_map"]  # [1,256,H/8,W/8]
        pre_feat = output["prev_feat"]
    kpts_resized = get_keypoints_from_semi_no_nms(semi)
    descriptors = extract_descriptors(desc, kpts_resized)
    if resize:
        scale_x = W0 / resize[1] #(240,320)
        scale_y = H0 / resize[0]
        kpts = np.zeros_like(kpts_resized)
        kpts[:, 0] = kpts_resized[:, 0] * scale_x  # x
        kpts[:, 1] = kpts_resized[:, 1] * scale_y  # y
        kpts[:, 2] = kpts_resized[:, 2]            # score
    else:
        kpts = kpts_resized

    return kpts, descriptors, pre_feat

def get_keypoints_from_semi_Ours(semi, threshold=0.005, nms_dist=0, top_k=300):
    heatmap = torch.softmax(semi, dim=1)[:, :-1, :, :]
    heatmap = heatmap.squeeze().cpu().numpy()
    Hc, Wc = heatmap.shape[1], heatmap.shape[2]
    prob = np.max(heatmap, axis=0)
    prob_idx = np.argmax(heatmap, axis=0)  # [Hc,Wc]
    keypoints = np.argwhere(prob > threshold)  # (y,x)
    if keypoints.shape[0] == 0:
        return np.zeros((0, 3))
    xs, ys = keypoints[:, 1], keypoints[:, 0]
    scores = prob[ys, xs]

    in_corners = np.vstack((xs, ys, scores))
    if nms_dist!=0 and keypoints.shape[0] > 100:
        nmsed_corners, _ = nms_fast(in_corners, Hc, Wc, nms_dist)
    else:
        nmsed_corners = in_corners

    pts_out = []
    for x, y, s in nmsed_corners.T:
        c = prob_idx[int(y), int(x)]
        x_orig = x * 8 + (c % 8)
        y_orig = y * 8 + (c // 8)
        pts_out.append([x_orig, y_orig, s])

    return np.array(pts_out)
def get_keypoints_from_semi_no_nms(semi, threshold=0.005):
    heatmap = torch.softmax(semi, dim=1)[:, :-1, :, :]
    heatmap = heatmap.squeeze().cpu().numpy()
    Hc, Wc = heatmap.shape[1], heatmap.shape[2]
    prob = np.max(heatmap, axis=0)
    prob_idx = np.argmax(heatmap, axis=0)  # [Hc,Wc]
    keypoints = np.argwhere(prob > threshold)  # (y,x)
    if keypoints.shape[0] == 0:
        return np.zeros((0, 3))
    xs, ys = keypoints[:, 1], keypoints[:, 0]
    scores = prob[ys, xs]
    in_corners = np.vstack((xs, ys, scores))
    pts_out = []
    for x, y, s in in_corners.T:
        c = prob_idx[int(y), int(x)]
        x_orig = x * 8 + (c % 8)
        y_orig = y * 8 + (c // 8)
        pts_out.append([x_orig, y_orig, s])

    return np.array(pts_out)
def nms_fast(in_corners, H, W, dist_thresh):
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
        return out, np.zeros((1)).astype(int)
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[1,i], rcorners[0,i]] = 1
        inds[rcorners[1,i], rcorners[0,i]] = i
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad),(pad,pad)), mode='constant')
    for i, rc in enumerate(rcorners.T):
        pt = (rc[0]+pad, rc[1]+pad)
        if grid[pt[1], pt[0]] == 1:
            grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
            grid[pt[1], pt[0]] = -1
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy-pad, keepx-pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds

def extract_descriptors(desc, kpts):
    desc = desc.squeeze().cpu().numpy()  # [256,Hc,Wc]
    Hc, Wc = desc.shape[1], desc.shape[2]
    descriptors = []
    for x, y, _ in kpts:
        xc, yc = int(x//8), int(y//8)
        xc = np.clip(xc, 0, Wc-1)
        yc = np.clip(yc, 0, Hc-1)
        d = desc[:, yc, xc]
        descriptors.append(d)
    descriptors = np.array(descriptors)
    return descriptors

def superpoint_desc_to_32byte(descriptors: np.ndarray) -> np.ndarray:
    if descriptors is None or descriptors.size == 0:
        return np.zeros((0, 32), dtype=np.uint8)

    descriptors = descriptors.astype(np.float32)
    norm = np.linalg.norm(descriptors, axis=1, keepdims=True) + 1e-8
    desc_n = descriptors / norm
    thr = desc_n.mean(axis=1, keepdims=True)
    bits = (desc_n > thr).astype(np.uint8)
    orb_like = np.packbits(bits, axis=1)
    return orb_like


def nms_matches_ORB(matches, cur_vkp, radius=12):
    selected = []
    occupied = np.zeros((len(matches),), dtype=bool)

    for i, m in enumerate(matches):
        if occupied[i]:
            continue
        pt1 = np.array(cur_vkp[m.trainIdx].pt)
        selected.append(m)
        for j in range(i + 1, len(matches)):
            if occupied[j]:
                continue
            pt2 = np.array(cur_vkp[matches[j].trainIdx].pt)
            if np.linalg.norm(pt1 - pt2) < radius:
                occupied[j] = True
    return selected

def compute_matching_accuracy(kp1, des1, kp2, des2, true_matrix=None, method='H', ratio_test=0.75, matches = None):
    if matches is None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

        if ratio_test:
            raw_matches = bf.knnMatch(des1, des2, k=2)
            matches = []
            for m, n in raw_matches:
                if m.distance < ratio_test * n.distance:
                    matches.append(m)
        else:
            matches = bf.match(des1, des2)

        if len(matches) == 0:
            return [], 0.0, None
    elif len(matches) == 0:
            return [], 0.0, None

    pts1 = np.float32([kp1[m.queryIdx][:2] for m in matches])  # [x,y]
    pts2 = np.float32([kp2[m.trainIdx][:2] for m in matches])

    if true_matrix is not None:
        if method == 'H':
            pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))]).T  # 3xN
            pts2_proj_h = (true_matrix @ pts1_h).T
            pts2_proj = pts2_proj_h[:, :2] / pts2_proj_h[:, 2:3]
            errors = np.linalg.norm(pts2 - pts2_proj, axis=1)
            mask = (errors < 3.0).astype(np.uint8)  # 内点阈值，可调
        elif method == 'F':
            pts1_h = np.hstack([pts1, np.ones((pts1.shape[0], 1))])
            pts2_h = np.hstack([pts2, np.ones((pts2.shape[0], 1))])
            lines1 = (true_matrix.T @ pts2_h.T).T  # l = F^T x'
            errors = np.abs(np.sum(lines1 * pts1_h, axis=1)) / np.linalg.norm(lines1[:, :2], axis=1)
            mask = (errors < 1.0).astype(np.uint8)  # 内点阈值，可调
        else:
            raise ValueError("method must be 'H' or 'F'")
    else:
        if method == 'H':
            _, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 3.0)
        elif method == 'F':
            if len(matches) < 7:
                return [], 0.0, None
            _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 1.0, confidence=0.99, maxIters=2000)
            mask = mask.ravel()
        else:
            raise ValueError("method must be 'H' or 'F'")

    mask = (mask > 0).astype(np.uint8)  # 确保 0/1
    accuracy = np.sum(mask) / len(matches)
    if accuracy > 1:
        print("accuracy > 1, and np.sum(mask) is ", np.sum(mask), " len(matches) is", len(matches))
    return matches, accuracy, mask
