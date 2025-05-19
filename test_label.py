import math

import torch
import xlrd
from torch import exp
#
# from load_data import get_loader
#
# dataset = 'mirflickr'
# DATA_DIR = 'data/' + dataset + '/'
# batch_size = 5
#
#
# # def calculate_adj(label_matrix, tau=1):
# #     a = torch.mm(label_matrix, label_matrix.t())
# #     a = a.int()
# #     adj = torch.ones(size=(label_matrix.size(0), label_matrix.size(0)))
# #
# #     for i in range(label_matrix.size(0)):
# #         for j in range(label_matrix.size(0)):
# #             if i != j:
# #                 adj[i, j] = 1 - exp(tau * torch.neg(a[i, j]))
# #
# #     adj1 = torch.cat([adj, adj], dim=0)
# #     adj2 = torch.cat([adj1, adj1], dim=1)
# #     return adj2
#
#
# if __name__ == '__main__':
#
#     data_loader, input_data_par = get_loader(DATA_DIR, batch_size)
#     i = 0
#     for imgs, txts, labels in data_loader['train']:
#         if i == 1:
#             break
#         print(calculate_adj(labels))
#         i = i+1
# import matplotlib.pyplot as plt
#
# # 示例数据
# y = [2, 4, 6, 2, 12]
#
# # 使用 plot 函数绘制折线图
# plt.plot( y)
#
# # 显示图形
# plt.show()



# def getLen(data_loaders):
#     train_size = len(data_loaders)
#     return train_size

# sheet = xlrd.open_workbook('./data/inflectionPoint/codetable.xlsx').sheet_by_index(0)
# # threshold = sheet.row(64)[math.ceil(math.log(24, 2))].value
# print(math.ceil(math.log(80, 2)))
# threshold = sheet.row(16)[7].value
#
# print("threshold:"+str(threshold))
# x = torch.tensor([[1, 2],
#                   [3, 4],
#                   [5, 6]])
# y = torch.tensor([[1, 2],
#                   [3, 4],
#                   [5, 6]])
# print(x)
# print(y)
# x = x.unsqueeze(1)
# y = y.unsqueeze(0)
# print(x-y)
# print((x-y).shape)
# z = x-y
# z = z.reshape(3 * 3, 2)
# print(z)
# print(z.shape)
#
# x = torch.tensor([1, 2, 3, 4, 5, 6])
# x = x.reshape(3, 2)
# print(x)




# import numpy as np
#
# def multilabel_accuracy(predict, label):
#     """
#     计算多标签分类的准确性
#     参数:
#     - predict: 预测矩阵，形状为 (n_samples, n_labels)
#     - label: 标签矩阵，形状为 (n_samples, n_labels)
#
#     返回:
#     - accuracy: 准确性，标量值
#     """
#     # 确保输入是numpy数组
#     predict = np.array(predict)
#     label = np.array(label)
#
#     # 检查预测矩阵和标签矩阵的形状是否匹配
#     assert predict.shape == label.shape, "预测矩阵和标签矩阵的形状必须一致"
#
#     # 计算每个样本的标签是否完全匹配
#     exact_match = np.all(predict == label, axis=1)
#
#     # 计算准确性
#     accuracy = np.mean(exact_match)
#
#     return accuracy
#
# # 示例使用
# predict = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
# label = np.array([[1, 0, 1], [1, 1, 0], [1, 1, 1]])
#
# accuracy = multilabel_accuracy(predict, label)
# print(f"多标签分类的准确性: {accuracy:.2f}")

import torch

a = torch.tensor([[[[1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1]],
                  ],
                 [[[1, 1, 1], [1, 1, 1]],
                  [[1, 1, 1], [1, 1, 1]],
                  ]])
print(a.size())