import pickle
import torch
import numpy as np
import os
import pandas as pd


def add_noise_to_data(images, noise_rate, noise_std):

    # 计算需要添加噪声的样本数
    n = images.shape[0]
    num_noisy_samples = int(n * noise_rate)

    # 随机选择 num_noisy_samples 个索引
    noisy_indices = np.random.choice(n, num_noisy_samples, replace=False)

    # 生成高斯噪声
    noise = np.random.randn(num_noisy_samples, images.shape[1]) * noise_std

    # 添加噪声
    if isinstance(images, torch.Tensor):
        images = images.clone()  # 避免修改原始数据
        images[noisy_indices] += torch.tensor(noise, dtype=images.dtype, device=images.device)
    else:
        images = images.copy()
        images[noisy_indices] += noise

    return images


def generate_noise_F(noise, noise_std):
    noise_rate = noise
    for i in noise_rate:
        with open('data/mirflickr/train.pkl', 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_texts = torch.tensor(data['text'], dtype=torch.float32)
            train_images = torch.tensor(data['image'], dtype=torch.float32)
        images = add_noise_to_data(train_images, i, noise_std)
        noise_images_list = np.array(images, dtype=np.float64)
        output_file = os.path.join('data/mirflickr/noise/', 'mirflickr25k-iall-noise_{}.pkl'.format(i))
        with open(output_file, 'wb') as f:
            pickle.dump(noise_images_list, f)

        texts = add_noise_to_data(train_texts, i, noise_std)
        noise_texts_list = np.array(texts, dtype=np.float64)
        output_file = os.path.join('data/mirflickr/noise/', 'mirflickr25k-tall-noise_{}.pkl'.format(i))
        with open(output_file, 'wb') as f:
            pickle.dump(noise_texts_list, f)


def generate_noise_M(noise, noise_std):
    noise_rate = noise
    for i in noise_rate:
        with open('data/MS-COCO/train.pkl', 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_texts = torch.tensor(data['text'], dtype=torch.float32)
            train_images = torch.tensor(data['image'], dtype=torch.float32)
        images = add_noise_to_data(train_images, i, noise_std)
        noise_images_list = np.array(images, dtype=np.float64)
        output_file = os.path.join('data/MS-COCO/noise/', 'COCO-iall-noise_{}.pkl'.format(i))
        with open(output_file, 'wb') as f:
            pickle.dump(noise_images_list, f)

        texts = add_noise_to_data(train_texts, i, noise_std)
        noise_texts_list = np.array(texts, dtype=np.float64)
        output_file = os.path.join('data/MS-COCO/noise/', 'COCO-tall-noise_{}.pkl'.format(i))
        with open(output_file, 'wb') as f:
            pickle.dump(noise_texts_list, f)

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
    generate_noise_M(noise_rate, 1)

    # file_path = 'data/mirflickr/noise/mirflickr25k-tall-noise_0.2.pkl'
    # file_path2 = './data/mirflickr/query.pkl'
    # data = load_pkl_file(file_path)
    # display_content(data)