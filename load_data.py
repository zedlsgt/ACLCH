import pickle

import torch
from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
from util import BackgroundGenerator
import hdf5storage
import numpy as np


class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class CustomDataSet(Dataset):
    def __init__(self, images, texts, labs):
        self.images = images
        self.texts = texts
        self.labs = labs

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        return img, text, lab

    def __len__(self):
        count = len(self.texts)
        return count


class CustomDataSet2(Dataset):
    def __init__(
            self,
            images_stu,
            texts_stu,
            labels,
            images,
            texts,
            multiLabels):
        self.images_stu = images_stu
        self.texts_stu = texts_stu
        self.labels = labels
        self.images = images
        self.texts = texts
        self.multiLabels = multiLabels

    def __getitem__(self, index):
        images_stu = self.images_stu[index]
        texts_stu = self.texts_stu[index]
        label = self.labels[index]
        img = self.images[index]
        text = self.texts[index]
        multiLabels = self.multiLabels[index]
        return images_stu, texts_stu, label, img, text, multiLabels

    def __len__(self):
        count = len(self.images)
        return count


def get_loader(path, batch_size, train_shuffle, noise, noise_data):
    # img_train = hdf5storage.loadmat(path + "train_img.mat")['train_img']
    # text_train = hdf5storage.loadmat(path + "train_txt.mat")['train_txt']
    # label_train = hdf5storage.loadmat(path + "train_lab.mat")['train_lab']

    train_path = path+'/train.pkl'
    query_path = path + '/query.pkl'
    retrieval_path = path + '/retrieval.pkl'
    if noise == 0 and noise_data == 0 :
        with open(train_path, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_labels = torch.tensor(data['label'], dtype=torch.int64)
            train_texts = torch.tensor(data['text'], dtype=torch.float32)
            train_images = torch.tensor(data['image'], dtype=torch.float32)
    elif noise > 0:
        with open(train_path, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_texts = torch.tensor(data['text'], dtype=torch.float32)
            train_images = torch.tensor(data['image'], dtype=torch.float32)
        with open(path+'noise/COCO-lall-noise_{}.pkl'.format(noise), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_labels = torch.tensor(data, dtype=torch.int64)
    elif noise_data > 0:
        with open(train_path, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_labels = torch.tensor(data['label'], dtype=torch.int64)
        with open(path+'noise/COCO-iall-noise_{}.pkl'.format(noise_data), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_images = torch.tensor(data, dtype=torch.float32)
        with open(path+'noise/COCO-tall-noise_{}.pkl'.format(noise_data), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_texts = torch.tensor(data, dtype=torch.float32)


    with open(query_path, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        query_labels = torch.tensor(data['label'], dtype=torch.int64)
        query_texts = torch.tensor(data['text'], dtype=torch.float32)
        query_images = torch.tensor(data['image'], dtype=torch.float32)

    with open(retrieval_path, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        retrieval_labels = torch.tensor(data['label'], dtype=torch.int64)
        retrieval_texts = torch.tensor(data['text'], dtype=torch.float32)
        retrieval_images = torch.tensor(data['image'], dtype=torch.float32)

    imgs = {'train': train_images, 'query': query_images, 'retrieval': retrieval_images}
    texts = {'train': train_texts, 'query': query_texts, 'retrieval': retrieval_texts}
    labs = {'train': train_labels, 'query': query_labels, 'retrieval': retrieval_labels}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x]) for x in ['train', 'query', 'retrieval']}
    shuffle = {'train': train_shuffle, 'query': False, 'retrieval': False}
    dataloader = {x: DataLoader(dataset[x], batch_size=batch_size, drop_last=True, pin_memory=True, shuffle=shuffle[x],
                                num_workers=0) for x in ['train', 'query', 'retrieval']}

    img_dim = train_images.shape[1]
    text_dim = train_texts.shape[1]
    num_class = train_labels.shape[1]

    input_data_par = {}
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class

    return dataloader, input_data_par


def get_loader_kd(path, batch_size, img_feat, text_feat, multiLabel, noise, noise_data):

    train_path = path + '/train.pkl'

    if noise == 0 and noise_data == 0 :
        with open(train_path, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_labels = torch.tensor(data['label'], dtype=torch.int64)
            train_texts = torch.tensor(data['text'], dtype=torch.float32)
            train_images = torch.tensor(data['image'], dtype=torch.float32)
    elif noise > 0:
        with open(train_path, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_texts = torch.tensor(data['text'], dtype=torch.float32)
            train_images = torch.tensor(data['image'], dtype=torch.float32)
        with open(path+'noise/COCO-lall-noise_{}.pkl'.format(noise), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_labels = torch.tensor(data, dtype=torch.int64)
    elif noise_data > 0:
        with open(train_path, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_labels = torch.tensor(data['label'], dtype=torch.int64)
        with open(path+'noise/COCO-iall-noise_{}.pkl'.format(noise_data), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_images = torch.tensor(data, dtype=torch.float32)
        with open(path+'noise/COCO-tall-noise_{}.pkl'.format(noise_data), 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            train_texts = torch.tensor(data, dtype=torch.float32)



    img_feat = img_feat.numpy()
    text_feat = text_feat.numpy()
    multiLabel = multiLabel.numpy()
    dataset = CustomDataSet2(images_stu=train_images, texts_stu=train_texts, labels=train_labels, images=img_feat, texts=text_feat, multiLabels=multiLabel)

    dataloader = DataLoaderX(dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=0, drop_last=True)
    img_dim = train_images.shape[1]
    text_dim = train_texts.shape[1]
    num_class = train_labels.shape[1]

    input_data_par = {}
    input_data_par['img_dim'] = img_dim
    input_data_par['text_dim'] = text_dim
    input_data_par['num_class'] = num_class

    return dataloader,input_data_par

