import cv2
import torch
import scipy.misc
from io import BytesIO
import numpy as np
from tensorboardX.summary import Summary
import matplotlib.colors as cl

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
        
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

if __name__ == '__main__':
    img0 = torch.tensor([[[-10, 10], [0, 0]]]).float()
    print(img0.shape)
    print(flow2rgb(img0.numpy()))
