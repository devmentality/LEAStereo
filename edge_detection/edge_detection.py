import numpy as np
import cv2
import torch
import torch.nn.functional as F


def make_smoother_disp(disp_arr_batch: np.ndarray) -> np.ndarray:
    smoothed_batch = []
    for disp_arr in disp_arr_batch:
        smoothed_batch.append(cv2.medianBlur(cv2.medianBlur(disp_arr, 5), 5))
    return np.array(smoothed_batch)


def transform_disp(disp_arr, min_thr):
    disp_arr[disp_arr < min_thr] = np.nan
    disp_arr = (-1) * disp_arr
    in_min = np.nanmin(disp_arr)
    in_rng = np.nanmax(disp_arr) - np.nanmin(disp_arr)

    new_arr = (disp_arr - in_min) * 255 / in_rng
    new_arr[np.isnan(new_arr)] = 0
    
    return new_arr.astype(np.uint8)


def make_disp_edges(disp_arr: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(transform_disp(disp_arr, -900), 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(edges, kernel, iterations=1)


def y_grad(t_arr: torch.Tensor, device) -> torch.Tensor:
    t_y_kernel = torch.Tensor([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ]).to(device=device)

    t_y_kernel = torch.unsqueeze(torch.unsqueeze(t_y_kernel, dim=0), dim=0)
    t_arr = torch.unsqueeze(t_arr, dim=1).float()

    t_y_grad = F.conv2d(t_arr, t_y_kernel)
    return t_y_grad 


def x_grad(t_arr: torch.Tensor, device) -> torch.Tensor:
    t_x_kernel = torch.Tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ]).to(device=device)

    t_x_kernel = torch.unsqueeze(torch.unsqueeze(t_x_kernel, dim=0), dim=0)
    t_arr = torch.unsqueeze(t_arr, dim=1).float()

    t_x_grad = F.conv2d(t_arr, t_x_kernel)
    return t_x_grad 


def gradient_aware_loss(output, target, device):
    target = torch.from_numpy(make_smoother_disp(target.cpu().numpy())).to(device=device)
    out_x, out_y = x_grad(output, device), y_grad(output, device)
    t_x, t_y = x_grad(target, device), y_grad(target, device)

    return F.smooth_l1_loss(out_x, t_x, reduction='mean') + F.smooth_l1_loss(out_y, t_y, reduction='mean')