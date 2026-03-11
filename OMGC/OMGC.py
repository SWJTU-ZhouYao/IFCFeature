import torch
import cv2, os
import numpy as np
import scipy.spatial as spatial


def rotation_optimization(kp, matches, pre_kp):
    rotHist = [[] for _ in range(30)]
    factor = 30 / 360.0
    for i, m in enumerate(matches):
        rot = pre_kp[m.queryIdx].angle - kp[m.trainIdx].angle
        if rot < 0.0:
            rot += 360.0
        bin_idx = round(rot * factor)
        if bin_idx == 30:
            bin_idx = 0
        rotHist[bin_idx].append(i)

    ind1, ind2, ind3 = compute_three_maxima(rotHist, 30)

    filtered_matches = []
    for i in [ind1, ind2, ind3]:
        if i != -1:
            for j in rotHist[i]:
                filtered_matches.append(matches[j])
    return filtered_matches

def geometry_optimizer(matches):
    good_matches = []
    min_dist = float('inf')
    max_dist = 0.0
    for match in matches:
        dist = match.distance
        if dist < min_dist:
            min_dist = dist
        if dist > max_dist:
            max_dist = dist
    if min_dist == 0 or max_dist == 0:
        distance_proportion = 1
    else:
        distance_proportion = max_dist / min_dist
    if distance_proportion >= 35:
        temp_threshold = min(35 * min_dist, max_dist / 2.5)
    else:
        temp_threshold = (min_dist + max_dist) / 2
    for match in matches:
        if match.distance <= temp_threshold:
            good_matches.append(match)
    return good_matches

def affine_scope_fast(kp, matches, pre_kp, dis_threshold=1600):
    kp_pts = np.array([k.pt for k in kp])
    pre_kp_pts = np.array([k.pt for k in pre_kp])
    Matching_indices = np.full(len(kp), -1, dtype=int)
    for m in matches:
        Matching_indices[m.trainIdx] = m.queryIdx
    tree_kp = spatial.cKDTree(kp_pts)
    tree_pre = spatial.cKDTree(pre_kp_pts)

    good_match_num = len(matches)
    use_num = max(4, 10 if good_match_num > 1000 else max(1, round(good_match_num * 0.01)))

    affine_matches = []

    for m in matches:
        p1 = pre_kp_pts[m.queryIdx]
        p2 = kp_pts[m.trainIdx]
        if dis_threshold > 0:
            idx_1 = tree_pre.query_ball_point(p1, np.sqrt(dis_threshold))
        else:
            idx_1 = np.arange(len(pre_kp_pts))

        if len(idx_1) > use_num:
            dists = np.sum((pre_kp_pts[idx_1] - p1)**2, axis=1)
            idx_1 = np.array(idx_1)[np.argsort(dists)[:use_num]]
        if dis_threshold > 0:
            idx_2 = tree_kp.query_ball_point(p2, np.sqrt(dis_threshold))
        else:
            idx_2 = np.arange(len(kp_pts))

        if len(idx_2) > use_num:
            dists = np.sum((kp_pts[idx_2] - p2)**2, axis=1)
            idx_2 = np.array(idx_2)[np.argsort(dists)[:use_num]]
        matching_pre = Matching_indices[idx_2]
        true_num = np.sum(np.isin(matching_pre, idx_1))

        affine_false_t = (len(idx_2) // 2) if (use_num != len(idx_2)) else (use_num // 2)
        if true_num >= affine_false_t:
            affine_matches.append(m)

    return affine_matches

def compute_three_maxima(hist, L):
    max1 = max2 = max3 = 0
    ind1 = ind2 = ind3 = -1

    for i in range(L):
        s = len(hist[i])
        if s > max1:
            max3, max2, max1 = max2, max1, s
            ind3, ind2, ind1 = ind2, ind1, i
        elif s > max2:
            max3, max2 = max2, s
            ind3, ind2 = ind2, i
        elif s > max3:
            max3 = s
            ind3 = i

    if max2 < 0.1 * max1:
        ind2, ind3 = -1, -1
    elif max3 < 0.1 * max1:
        ind3 = -1
    return ind1, ind2, ind3

def out_semi_desc_from_orb(cur_kp, cur_des):
    semi = np.zeros((1, 65, 30, 40), dtype=np.float32)
    desc = np.zeros((1, 256, 30, 40), dtype=np.float32)
    for index, kp in enumerate(cur_kp):
        x, y = kp.pt
        c = int(x % 8) + int(y % 8) * 8
        cx = int(x // 8)  # +c%8
        cy = int(y // 8)  # +c//8
        if max(semi[0, :, cy, cx]) < kp.response * 100:
            desc[0, :, cy, cx] = cur_des[index]
        semi[0, c, cy, cx] = kp.response * 100.0
    mask = np.any(semi[:, :64, :, :] != 0, axis=1)  # shape = (1, 30, 40)
    semi[:, 64, :, :] = np.where(mask, 0, 1)
    return torch.from_numpy(semi), torch.from_numpy(desc)

if __name__ == "__main__":
    resize = (240, 320)
    orb = cv2.ORB_create(2500, 1.2, 8, 31, 0, 2,
                         cv2.ORB_HARRIS_SCORE, 31, 7)
    indices = ['MH_01_easy','MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult','V1_01_easy', 'V1_02_medium', 'V1_03_difficult','V2_01_easy', 'V2_02_medium', 'V2_03_difficult'] #
    orin_name = "image0_orb_Rota_Geo_Affine_Repro_npz"
    for i in indices:
        img_folder = ""
        if not os.path.exists(img_folder + '/' + orin_name):
            os.makedirs(img_folder + '/' + orin_name)
            print(f"文件夹 '{img_folder + '/' + orin_name}' Creat")
        else:
            print(f"文件夹 '{img_folder + '/' + orin_name}' Existed")

        img_files = sorted([f for f in os.listdir(img_folder + '/cam0/data') if
                            f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
        img_base_path = img_folder + '/cam0/data'

        ini_img = cv2.imread(os.path.join(img_base_path, img_files[0]), cv2.IMREAD_GRAYSCALE)
        ini_img_resized = cv2.resize(ini_img, (resize[1], resize[0]))
        pre_kp, pre_des = orb.detectAndCompute(ini_img_resized, None)
        pre_des = np.unpackbits(pre_des, axis=1).astype(np.float32)
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        for img_path in img_files:
            orin_save_path = img_folder + '/' + orin_name + '/' + str(img_path[0:-4]) + ".npz"
            img_path = os.path.join(img_base_path, img_path)
            cur_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            cur_img_resized = cv2.resize(cur_img, (resize[1], resize[0]))
            cur_kp, cur_des = orb.detectAndCompute(cur_img_resized, None)
            cur_des = np.unpackbits(cur_des, axis=1).astype(np.float32)
            matches = bf.match(pre_des, cur_des)
            if len(matches) > 400:
                rota_matches = rotation_optimization(cur_kp, matches, pre_kp)
                if len(rota_matches) > 300:
                    matches = rota_matches
            if len(matches) > 300:
                geo_matches = geometry_optimizer(matches)
                if len(geo_matches) > 250:
                    matches = geo_matches
            if len(matches) > 250:
                affine_matches = affine_scope_fast(cur_kp, matches, pre_kp, 400)
                if len(affine_matches) > 200:
                    matches = affine_matches

            tem_cur_kp = []
            tem_cur_des = []
            for match in matches:
                tem_cur_kp.append(cur_kp[match.trainIdx])
                tem_cur_des.append(cur_des[match.trainIdx])
            false_point_in_cur = []
            semi, desc = out_semi_desc_from_orb(tem_cur_kp, tem_cur_des)

            if len(matches) > 200:
                pts1 = np.float32([pre_kp[m.queryIdx].pt for m in matches])
                pts2 = np.float32([cur_kp[m.trainIdx].pt for m in matches])
                _, mask = cv2.findFundamentalMat(pts1, pts2, cv2.RANSAC, 1.0, confidence=0.99, maxIters=2000)
                inlier_matches = [m for m, inlier in zip(matches, mask.ravel()) if inlier]
                if len(inlier_matches) > 150:
                    matches = inlier_matches
            tem_cur_kp = []
            tem_cur_des = []
            for match in matches:
                tem_cur_kp.append(cur_kp[match.trainIdx])
                tem_cur_des.append(cur_des[match.trainIdx])
            orin_semi, orin_desc = out_semi_desc_from_orb(tem_cur_kp, tem_cur_des)

            pre_kp = np.asarray(cur_kp)
            pre_des = np.asarray(cur_des)
            np.savez(orin_save_path, semi=orin_semi, desc=orin_desc, false_point_in_cur=false_point_in_cur)





