#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
@author: Guweiwei, Gongli
@function: provide two algorithm to align two points sets
'''

import math
import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import fmin_l_bfgs_b

class Point_Align(object):
    def __init__(self, base_pos, pos):
        self.base_pos = base_pos
        self.pos = pos

    # rescale the data
    def rescale(self):
        self.pos *= np.sqrt((self.base_pos ** 2).sum()) / np.sqrt((self.pos ** 2).sum())

    #　用pca方法对齐
    def pca_align(self):
        clf = PCA(n_components=2)
        self.base_pos = clf.fit_transform(self.base_pos)
        self.pos = clf.fit_transform(self.pos)

    # 用优化算法对齐
    def optimize_align(self):
        # 定义旋转图形函数表示
        def rotate(original_pos, x, y, z):
            # x = 沿x轴的平移量
            # y = 沿y轴的平移量
            # z = 逆时针方向旋转的角度
            rotation = np.array([[math.cos(z), math.sin(z)], [-math.sin(z), math.cos(z)]])  # 坐标的旋转矩阵
            distance = np.array([x,y])  # 平移向量
            return np.dot(original_pos, rotation) + distance  # 旋转平移后的坐标

        # 定义优化函数
        def optimize_function(params):
            x, y, z = params
            # x = 沿x轴的平移量
            # y = 沿y轴的平移量
            # z = 逆时针方向旋转的角度
            rotated_pos = rotate(self.pos, x, y, z)  # 计算旋转后坐标
            v = []  # 计算欧氏距离（基准与图形变换后）
            for i in range(len(self.base_pos)):
                m = self.base_pos[i, :]
                n = rotated_pos[i, :]
                d = np.dot(n - m, n - m)
                v.append(np.sqrt(d))
            return sum(v)  # 得到该年的待优化函数

        initial_values = np.array([0, 0, 0])
        mybounds = [(None, None), (None, None), (None, None)]
        f = fmin_l_bfgs_b(optimize_function, x0 = initial_values, bounds = mybounds, approx_grad = True)
        self.pos = rotate(self.pos, f[0][0], f[0][1], f[0][2])

    def align(self):
        self.rescale()
        self.pca_align()
        self.optimize_align()
        return self.base_pos, self.pos
