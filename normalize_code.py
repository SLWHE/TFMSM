'''
import numpy as np
import torch


def normalize_matrix(matrix1, matrix2):
    """
    对矩阵进行归一化处理
    :param matrix: 输入的二维NumPy数组
    :return: 归一化后的矩阵, 每行的最小值数组, 每行的最大值数组
    """
    matirx_v = np.row_stack((matrix1, matrix2))
    mid_matirx = np.concatenate((matrix1, matrix2), axis=-1)
    # 第一步：每一行减去该行的最小值
    row_min = np.min(mid_matirx, axis=2, keepdims=True)
    midvar = mid_matirx - row_min
    row_min_all = np.row_stack((row_min, row_min))
    midvar_all = matirx_v - row_min_all

    # 第二步：每一行除以该行的最大值
    row_max = np.max(np.abs(midvar), axis=2, keepdims=True)
    # 避免除以0的情况，如果某行最大值为0，则该行的归一化结果全为0
    row_max[row_max == 0] = 1  # 这样处理可以避免除以0，但会使得该行归一化后全为0（因为midvar全为0）
    row_max_all = np.row_stack((row_max, row_max))
    normalized_matrix = midvar_all / row_max_all

    return normalized_matrix, row_min, row_max

def denormalize_matrix(normalized_matrix, row_min, row_max):
    """
    对归一化后的矩阵进行反归一化
    :param normalized_matrix: 归一化后的矩阵
    :param row_min: 每行的最小值数组
    :param row_max: 每行的最大值数组
    :return: 反归一化后的矩阵
    """
    # 反归一化：normalized_matrix * row_max + row_min
    denormalized_matrix = normalized_matrix * row_max + row_min
    return denormalized_matrix

'''
import torch


def normalize_matrix(matrix1, matrix2):
    """
    对矩阵进行归一化处理（GPU版本）
    :param matrix1: 输入的二维PyTorch张量
    :param matrix2: 输入的二维PyTorch张量
    :param device: 计算设备，默认为matrix1所在设备
    :return: 归一化后的矩阵, 每行的最小值数组, 每行的最大值数组
    """

    # 堆叠矩阵（替换numpy的row_stack和concatenate）
    matrix_v = torch.cat((matrix1, matrix2), dim=1)
    mid_matrix = torch.cat((matrix1, matrix2), dim=-1)

    # 第一步：每一行减去该行的最小值
    row_min = torch.min(mid_matrix, dim=-1, keepdim=True)[0]  # [0]获取最小值，[1]是索引
    midvar = mid_matrix - row_min
    row_min_all = torch.cat((row_min, row_min), dim=1)
    midvar_all = matrix_v - row_min_all

    # 第二步：每一行除以该行的最大值
    row_max = torch.max(torch.abs(midvar), dim=-1, keepdim=True)[0]
    # 避免除以0的情况
    row_max = torch.where(row_max == 0, torch.tensor(1.0, device='cuda'), row_max)
    row_max_all = torch.cat((row_max, row_max), dim=1)
    normalized_matrix = midvar_all / row_max_all

    return normalized_matrix, row_min, row_max


def denormalize_matrix(normalized_matrix, row_min, row_max):
    """
    对归一化后的矩阵进行反归一化（GPU版本）
    :param normalized_matrix: 归一化后的PyTorch张量
    :param row_min: 每行的最小值数组（PyTorch张量）
    :param row_max: 每行的最大值数组（PyTorch张量）
    :return: 反归一化后的矩阵（PyTorch张量）
    """
    # 确保所有张量在同一设备
    device = normalized_matrix.device
    # 反归一化
    denormalized_matrix = normalized_matrix * row_max + row_min
    return denormalized_matrix
