import os
import pickle
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm


def add_noise_to_labels(labels, noise_rate):
    num_samples, num_labels = labels.shape
    num_noise = int(num_samples * noise_rate)
    # 创建一个随机的索引列表，用于选择要添加噪声的样本
    noise_indices = np.random.choice(num_samples, num_noise, replace=False)
    # 随机改变选定样本的标签
    for i in tqdm(noise_indices):
        ones_indices = np.where(labels[i, :] == 1)[0]
        zeros_indices = np.where(labels[i, :] == 0)[0]
        # 随机选择一个值为1的元素，并将其变为0
        if len(ones_indices) > 0:
            j = np.random.choice(ones_indices)
            labels[i, j] = 0

        # 随机选择一个值为0的元素，并将其变为1
        if len(zeros_indices) > 0:
            j = np.random.choice(zeros_indices)
            labels[i, j] = 1
    return labels


def generate_noise_F(noise):
    noise_rate = noise
    for i in noise_rate:

        with open('data/mirflickr/train.pkl', 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            # train_labels = torch.tensor(data['label'], dtype=torch.int64)

        noisy_labels_matrix = add_noise_to_labels(np.array(data['label']), i)
        noisy_labels_matrix = np.array(noisy_labels_matrix, dtype=np.uint8)
        output_file = os.path.join('data/mirflickr/noise/', 'mirflickr25k-lall-noise_{}.pkl'.format(i))

        with open(output_file, 'wb') as f:
            pickle.dump(noisy_labels_matrix, f)



def generate_noise_N(noise):
    noise_rate = noise
    data = h5py.File('./data/NUS-WIDE.h5', 'r')
    for i in noise_rate:
        # 添加噪声
        labels_matrix = np.array(list(data['LabTrain']))
        labels_matrix2 = np.array(list(data['LabTrain']))
        noisy_labels_matrix = add_noise_to_labels(labels_matrix, i)

        output_file = h5py.File('./noise/nus-wide-tc21-lall-noise_{}.h5'.format(i), 'w')

        # 将模型输出和损失保存到.h5文件中
        output_file.create_dataset('result', data=noisy_labels_matrix)
        output_file.create_dataset('True', data=labels_matrix2)

        # 关闭.h5文件
        output_file.close()


def generate_noise_M(noise):
    noise_rate = noise
    for i in noise_rate:

        with open('data/MS-COCO/train.pkl', 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            # train_labels = torch.tensor(data['label'], dtype=torch.int64)

        noisy_labels_matrix = add_noise_to_labels(np.array(data['label']), i)
        noisy_labels_matrix = np.array(noisy_labels_matrix, dtype=np.uint8)
        output_file = os.path.join('data/MS-COCO/noise/', 'COCO-lall-noise_{}.pkl'.format(i))

        with open(output_file, 'wb') as f:
            pickle.dump(noisy_labels_matrix, f)


def load_pkl_file(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
        return data


def display_content(data):
    if isinstance(data, pd.DataFrame):
        print("数据框内容:")
        print(data.head())
    else:
        print("文件内容:")
        print(data)


if __name__ == "__main__":
    noise_rate = [0.2, 0.4, 0.6, 0.8]
    generate_noise_M(noise_rate)

    # file_path = 'data/mirflickr/noise/mirflickr25k-lall-noise_0.2.pkl'
    # file_path2 = './data/mirflickr/query.pkl'
    # data = load_pkl_file(file_path2)
    # display_content(data)