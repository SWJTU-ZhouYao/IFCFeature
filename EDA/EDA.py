import os
import cv2, time
import numpy as np
from tqdm import tqdm
import math
import matplotlib.pyplot as plt

def nms_keypoints(keypoints, descriptors, dist_thresh=8):
    if not keypoints:
        return [], []

    sorted_idx = np.argsort([-kp.response for kp in keypoints])
    keypoints = [keypoints[i] for i in sorted_idx]
    descriptors = descriptors[sorted_idx, :]

    keep_kp = []
    keep_des = []

    suppressed = np.zeros(len(keypoints), dtype=bool)

    for i, kp_i in enumerate(keypoints):
        if suppressed[i]:
            continue
        keep_kp.append(kp_i)
        keep_des.append(descriptors[i])

        x_i, y_i = kp_i.pt

        for j in range(i + 1, len(keypoints)):
            if suppressed[j]:
                continue
            x_j, y_j = keypoints[j].pt
            if (x_i - x_j)**2 + (y_i - y_j)**2 < dist_thresh**2:
                suppressed[j] = True

    keep_des = np.stack(keep_des, axis=0)
    return keep_kp, keep_des

def ease_cosine(x):
    """平滑过渡 [0,1] → [0,1]"""
    return (1 - math.cos(x * math.pi)) / 2

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

def add_noise(img, std):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    noisy = np.clip(img.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)

def shift_color(img, ratio):
    factor = 1.0 - 0.2 * ratio
    return np.clip(img * factor, 0, 255).astype(np.uint8)

def add_flare(img, strength):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.circle(mask, (int(w*0.8), int(h*0.4)), int(w*0.5), 1, -1)
    mask = cv2.GaussianBlur(mask, (0,0), 100)
    flare = (img.astype(np.float32) * (1 + strength*mask)).clip(0, 255).astype(np.uint8)
    return flare

def compute_dark_light_ratio(t1, t2, t3, t4, frame):
    """返回 L(t) ∈ [0.4, 1.0]"""
    if frame < t1:
        return 1.0
    elif t1 <= frame < t2:
        r = ease_cosine((frame - t1)/(t2 - t1))
        return 1.0 - 0.6*r
    elif t2 <= frame < t3:
        return alpha_min
    elif t3 <= frame < t4:
        r = ease_cosine((frame - t3)/(t4 - t3))
        return alpha_min + (1.0 - alpha_min)*r
    else:
        return 1.0

def dark_light(t1, t2, t3, t4, frame, img):
    L = compute_dark_light_ratio(t1, t2, t3, t4, frame)
    alpha = L
    gamma = 1 + (gamma_max - 1) * (1 - L)
    beta = -30 * (1 - L)
    noise_std = noise_max * (1 - L)
    cold_ratio = 1 - L

    img_adj = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    img_gamma = adjust_gamma(img_adj, gamma)
    img_noisy = add_noise(img_gamma, noise_std)
    img_color = shift_color(img_noisy, cold_ratio)

    if t3 <= frame <= t4:
        img_color = add_flare(img_color, flare_strength * (1 - L))

    h, w = img.shape[:2]
    xv, yv = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    vignette = 1 - 0.4 * (xv ** 2 + yv ** 2)
    img_color = np.clip(img_color * vignette, 0, 255).astype(np.uint8)

    return img_color

BRIGHT_MIN = 1.0
BRIGHT_MAX = 1.6

def compute_bright_light_ratio(t1, t2, t3, t4, frame):
    if frame < t1:
        return BRIGHT_MIN
    elif t1 <= frame < t2:
        r = ease_cosine((frame - t1) / (t2 - t1))
        return BRIGHT_MIN + (BRIGHT_MAX - BRIGHT_MIN) * r
    elif t2 <= frame < t3:
        return BRIGHT_MAX
    elif t3 <= frame < t4:
        r = ease_cosine((frame - t3) / (t4 - t3))
        return BRIGHT_MAX - (BRIGHT_MAX - BRIGHT_MIN) * r
    else:
        return BRIGHT_MIN


def bright_light(t1, t2, t3, t4, frame, img, gamma_max=1.8, beta_max=35, noise_max=8.0, flare_strength=0.6,
                vignette_base=0.30, use_shift_color=False):
    L = compute_bright_light_ratio(t1, t2, t3, t4, frame)

    p = (L - BRIGHT_MIN) / (BRIGHT_MAX - BRIGHT_MIN)
    p = float(np.clip(p, 0.0, 1.0))

    alpha = L

    beta = beta_max * p

    gamma = 1.0 + (gamma_max - 1.0) * p

    noise_std = noise_max * (1.0 - p)

    img_adj   = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    img_gamma = adjust_gamma(img_adj, gamma)
    img_noisy = add_noise(img_gamma, noise_std)

    img_color = img_noisy

    if use_shift_color:
        ratio = 0.0
        img_color = shift_color(img_color, ratio)

    if t1 <= frame <= t3:
        img_color = add_flare(img_color, flare_strength * p)

    h, w = img.shape[:2]
    xv, yv = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    vignette_strength = vignette_base * (1.0 - p)
    vignette = 1.0 - vignette_strength * (xv**2 + yv**2)
    img_color = np.clip(img_color.astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    return img_color

if __name__ == "__main__":
    indices = ['MH_01_easy']
    for sequence_name in indices:
        print("Staring sequence：", sequence_name)
        folder_path = ""
        output_dir = ""
        os.makedirs(output_dir, exist_ok=True)
        file_names = os.listdir(folder_path)
        full_paths = [os.path.join(folder_path, f) for f in file_names]
        all_length = len(full_paths)

        dark_light_start = 0*all_length
        dark_light_end = 0.4*all_length
        strong_light_start = 0.6*all_length
        strong_light_end = 1*all_length
        interval = int(0.33*(strong_light_end - strong_light_start))
        alpha_min = 0.4
        gamma_max = 1.8
        noise_max = 10
        flare_strength = 1.2

        for i, fname in enumerate(tqdm(file_names)):
            img = cv2.imread(os.path.join(folder_path, fname), cv2.IMREAD_GRAYSCALE)
            frame = i + 1
            if dark_light_start <= frame <= dark_light_end:
                img_color = dark_light(dark_light_start, dark_light_start+interval, dark_light_start+2*interval, dark_light_end, frame, img)
            elif strong_light_start <= frame <= strong_light_end:
                img_color = bright_light(strong_light_start, strong_light_start+interval, strong_light_start+2*interval, strong_light_end, frame, img)
            else:
                img_color = img.copy()
            cv2.imwrite(os.path.join(output_dir, fname), img_color)
