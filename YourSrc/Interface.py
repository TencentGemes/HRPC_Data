import torch
import os
import numpy as np

from ExampleInference import inference

def reconstruct(dataset, lightmap_id, coords):
    # 在该函数中，参赛者需要根据输入的dataset, lightmap_id以及coords，返回对应点的lightmap值
    # dataset为数据集名称，请处理所有大赛给出的数据集: ['OldFactory', ...]
    # lightmap_id与数据集中lightmap_list中的id对应
    # coords为一个numpy数组，形状为[N, 3]，表示输入N个坐标点，每个坐标点有y, x, time三个分量
    # y对应lightmap纵坐标，x代表横坐标，并且y, x为非归一化的坐标，代表像素的坐标，如[0, 0]代表左上角像素，[0, 50]代表第一行的第51个像素
    # time代表时间，time为 0 <= time <= 24 的浮点数


    # 返回输入坐标对应的lightmap值，返回值为numpy数组，形状为[N, 3]，表示对应输入的N个坐标点，每个点有R, G, B三个分量，每个分量为float32类型
    return inference(dataset, lightmap_id, coords.astype(np.float32))