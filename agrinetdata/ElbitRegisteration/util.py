from pathlib import Path

import cv2
import numpy as np


def calculate_distance(depthImage, scale=0.2, window_size=30, use_elbit_eq=False):
    h, w = depthImage.shape[0:2]
    x1, y1, x2, y2 = int(w / 2 - w * scale - 1), int(h / 2 - h * scale - 1), int(w * scale * 2), int(h * scale * 2)
    croppedDepth = depthImage[y1:y2, x1:x2]
    histSize = 2 ** 16
    hist = cv2.calcHist([croppedDepth], [0], None, [histSize], [0, histSize], accumulate=True)
    hist = np.array(hist).reshape(hist.shape[0])
    hist_sorted_ind = np.argsort(hist)[::-1]
    hist_argmax = hist_sorted_ind[0] if hist_sorted_ind[0] != 0 else hist_sorted_ind[1]
    weighted_sum = 0.0
    aggregated_weight = 0.0
    for k in range(int(max(hist_argmax - window_size, 0)), int(min(histSize, hist_argmax + window_size))):
        weighted_sum += float(hist_sorted_ind[k]) * float(hist[k]) if use_elbit_eq else float(k) * float(hist[k])
        aggregated_weight += float(hist[k])
    return weighted_sum / aggregated_weight


def calculate_distance_old(depthImage, scale=0.1, top_k=30):
    h, w = depthImage.shape[0:2]
    x2, y2, x1, y1 = int(w / 2 - w * scale - 1), int(h / 2 - h * scale - 1), int(w * scale * 2), int(h * scale * 2)
    croppedDepth = depthImage[y1:y2, x1:x2]
    histSize = 2 ** 16
    hist = cv2.calcHist([croppedDepth], [0], None, [histSize], [0, histSize], accumulate=True)
    hist = np.array(hist).reshape(hist.shape[0])
    max_hist_idx = np.argsort(hist)[(-1 * top_k):][::-1]
    weighted_sum = 0.0
    aggregated_weight = 0.0
    for k in max_hist_idx:
        weighted_sum += k * float(hist[k])
        aggregated_weight += float(hist[k])
    return weighted_sum / aggregated_weight


if __name__ == '__main__':
    fpath = Path('../test/sample_data/3D_Channel_RealSense_Depth.tiff').absolute()
    img = cv2.imread(str(fpath), flags=cv2.IMREAD_UNCHANGED)
    dist = calculate_distance(img)
    print(dist)
