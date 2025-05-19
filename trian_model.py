from __future__ import print_function
from __future__ import division

import math

import hdf5storage
import torch
import torch.nn as nn
import torchvision
import time
import copy
import numpy as np
import torch.nn.functional as F
import xlrd
import matplotlib.pyplot as plt
from hdf5storage import loadmat
from torch import optim

from evaluate import fx_calc_map_label, calc_topk_precision
from load_data import get_loader, get_loader_kd
from model import TeaNN, StuNN
from util import gen_simples_adj, gen_simples_adj2

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)


def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim


def cal_loss_semantic(x, y, adj, threshold):

    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    cos_sim = F.cosine_similarity(x, y, dim=-1)
    pos = F.relu(1 - cos_sim - 0.2)
    neg = F.relu(cos_sim - threshold - 0.2)
    loss_neg = (torch.where(adj == 0, 1, 0) * neg).sum() / (
        torch.where(adj == 0, 1, 0)).sum()
    loss_pos = (torch.where(adj > 0, 1, 0) * adj * pos).sum() / (
        torch.where(adj > 0, 1, 0)).sum()
    return loss_neg+loss_pos


loss_l2 = nn.MSELoss()
criterion = nn.CrossEntropyLoss()
criterion_class = nn.MultiLabelSoftMarginLoss()


def cal_loss_semantic_discrepancy(x, y, m_label):
    cos_xx = torch.matmul(x, x.T)
    var_xx = torch.var(cos_xx, dim=1, keepdim=True)

    cos_yy = torch.matmul(y, y.T)
    var_yy = torch.var(cos_yy, dim=1, keepdim=True)
    omega_xy = var_xx.unsqueeze(1) + var_yy.unsqueeze(0)

    # var_xy = torch.matmul(var_xy,var_xy.T)
    # var_xy = var_xy.unsqueeze(-1)

    cos_mm = torch.matmul(m_label,m_label.T)
    var_mm = torch.var(cos_mm, dim=1, keepdim=True)
    # var_mm = torch.matmul(var_mm, var_mm.T)
    omega_mm = var_mm.unsqueeze(1) + var_mm.unsqueeze(0)
    # var_mm = var_mm.unsqueeze(-1)

    # x = x.unsqueeze(1)
    # y = y.unsqueeze(0)
    #
    m_label1 = m_label.unsqueeze(1)
    m_label0 = m_label.unsqueeze(0)
    # f = x-y
    # f = f**2
    #
    # feature_l2_norms = torch.norm(f, p=2, dim=2)

    # m = m_label1 - m_label0
    # m = m ** 2
    #
    # mlabel_l2_norms = torch.norm(m, p=2, dim=2)
    loss = torch.mean((omega_xy*(x - y) ** 2 - omega_mm*(m_label1 - m_label0) ** 2) ** 2)
    # loss = torch.mean(((x - y) ** 2 - (m_label1 - m_label0) ** 2) ** 2)
    # loss = torch.mean((feature_l2_norms - mlabel_l2_norms) ** 2)

    return loss


def calc_loss(view1_feature, view2_feature, m_label):

    # loss_q1 = loss_l2(torch.sign(view1_feature+view2_feature), view1_feature)
    # loss_q2 = loss_l2(torch.sign(view1_feature+view2_feature), view2_feature)
    loss_q1 = loss_l2(torch.sign(m_label), view1_feature)
    loss_q2 = loss_l2(torch.sign(m_label), view2_feature)
    loss_p1 = cal_loss_semantic_discrepancy(view1_feature, view1_feature, m_label)
    loss_p2 = cal_loss_semantic_discrepancy(view2_feature, view2_feature, m_label)
    loss_p3 = cal_loss_semantic_discrepancy(view1_feature, view2_feature, m_label)

    # 多标签分类损失
    # loss_mcla = torch.mean((view1_feature - m_label) ** 2) + torch.mean((view2_feature - m_label) ** 2)
    # loss_mcla = (((view1_feature - m_label.float()) ** 2).sum(1).sqrt().mean()
    #              + ((view2_feature - m_label.float()) ** 2).sum(1).sqrt().mean())

    loss_q = loss_q1 + loss_q2
    loss_p = loss_p1 + loss_p2 + loss_p3
    return loss_q, loss_p


def l2_norm(x):
    return torch.sqrt(torch.sum(x ** 2, dim=-1))


def calc_loss_stu(img_tea, text_tea, m_label, img_stu, text_stu):

    loss1 = cal_loss_semantic_discrepancy(img_stu,img_tea,m_label)
    loss2 = cal_loss_semantic_discrepancy(text_stu, text_tea, m_label)
    # loss3 = cal_loss_semantic_discrepancy(img_stu, text_tea, m_label)
    # loss4 = cal_loss_semantic_discrepancy(text_stu, img_tea, m_label)
    # loss1 = (l2_norm(img_stu - img_tea) - l2_norm(m_label1 - m_label0)) ** 2
    # loss2 = (l2_norm(text_stu - text_tea) - l2_norm(m_label1 - m_label0)) ** 2
    # loss3 = (l2_norm(img_stu - text_tea) - l2_norm(m_label1 - m_label0)) ** 2
    # loss4 = (l2_norm(text_stu - img_tea) - l2_norm(m_label1 - m_label0)) ** 2

    # loss_p = loss1 + loss2 + loss3 + loss4
    loss_p = loss1 + loss2
    # loss1 = loss_l2(view1_feature_stu, view1_feature)
    # loss2 = loss_l2(view2_feature_stu, view2_feature)
    return loss_p


def train_teacher(args):
    DATA_DIR = 'data/' + args.dataset + '/'

    if args.embedding == 'glove':
        inp = loadmat('embedding/' + args.dataset + '-inp-glove6B.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif args.embedding == 'googlenews':
        inp = loadmat('embedding/' + args.dataset + '-inp-googlenews.mat')['inp']
        inp = torch.FloatTensor(inp)
    elif args.embedding == 'fasttext':
        inp = loadmat('embedding/' + args.dataset + '-inp-fasttext.mat')['inp']
        inp = torch.FloatTensor(inp)
    else:
        inp = None
    print('...Data loading is beginning...')
    data_loader, input_data_par = get_loader(DATA_DIR, args.batch_size, train_shuffle=True, noise=args.noise, noise_data=args.noise_data)
    train_loader = data_loader['train']
    query_loader = data_loader['query']
    retrieval_loader = data_loader['retrieval']

    data_loader2, _ = get_loader(DATA_DIR, args.batch_size, train_shuffle=False, noise=args.noise, noise_data=args.noise_data)
    train_loader2 = data_loader2['train']
    print('...Data loading is completed...')
    print('...Training is beginning...')

    model_tea = TeaNN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                        num_classes=input_data_par['num_class'], inp=inp, batch_size=args.batch_size, hash_dim=args.hash_dim,
                        dropout=args.dropout).cuda()

    params_tea = model_tea.get_config_optim(args.lr)
    optimizer_G = optim.Adam(params_tea,lr=args.lr)
    # optimizer_D2 = optim.Adam(discriminator2.parameters(), lr=args.lr)
    since = time.time()
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_tea.train()
    training_loss_tea_results = []

    # sheet = xlrd.open_workbook('./data/inflectionPoint/codetable.xlsx').sheet_by_index(0)
    # threshold = sheet.row(args.hash_dim)[math.ceil(math.log(input_data_par['num_class'], 2))].value

    for epoch in range(args.tea_epoch):

        running_loss = 0.0
        running_loss_G =0.0
        # Iterate over data.
        for i,(imgs, txts, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels.cuda()
                simples_adj = gen_simples_adj(labels).cuda()
            else:
                simples_adj = gen_simples_adj(labels)

            # zero the parameter gradients
            optimizer_G.zero_grad()
            # Forward
            view1_feature, view2_feature, m_label, loss_cla, loss_cla_triplet, loss_simple_cla, adj_sf, adj_fs = model_tea(imgs, txts,
                                                                                                      simples_adj,
                                                                                                      labels, args.eta, args.lAmbda)
            fake1 = model_tea.discriminator1(adj_sf)
            real1 = model_tea.discriminator1(adj_fs)

            # real2 = model_tea.discriminator2(adj_sf)
            # fake2 = model_tea.discriminator2(adj_fs)
            # fake2 = discriminator2(adj_fs)

            # loss_G1 = loss_l2(fake1, torch.ones(fake1.shape[0], 1).cuda())
            loss_G1 = (criterion(fake1, torch.ones(fake1.shape[0], dtype=torch.long).cuda())
                       + criterion(real1, torch.zeros(real1.shape[0], dtype=torch.long).cuda()))
            # loss_G2 = (criterion(fake2, torch.ones(fake1.shape[0], dtype=torch.long).cuda())
            #            + criterion(real2,torch.zeros(real1.shape[0],dtype=torch.long).cuda()))
            loss_q, loss_p = calc_loss(view1_feature, view2_feature, m_label)

            loss = loss_G1 + args.alpha * loss_simple_cla + args.beta * (loss_cla + loss_cla_triplet) + args.gamma * loss_q + args.mu * loss_p

            # backward + optimize only if in training phase
            loss.backward()
            optimizer_G.step()
            running_loss += loss.item()
            running_loss_G += loss_G1.item()
            if i + 1 == len(train_loader) and (epoch+1) % 2 == 0:
                print('Epoch [%3d/%3d], Loss: %.4f, loss_G1: %.4f, loss_se: %.4f, loss_simple_class: %.4f, loss_q: %.4f, loss_cla: %.4f, loss_cla_triplet: %.4f, loss_p: %.5f'
                    % (epoch + 1, args.tea_epoch, loss.item(), loss_G1.item(), 0.0,
                       loss_simple_cla*args.alpha, loss_q.item()*args.gamma,loss_cla.item()*args.beta, loss_cla_triplet.item()*args.beta, args.mu * loss_p))

        # epoch_loss = running_loss
        # training_loss_tea_results.append(round(epoch_loss, 3))

        # print('Teacher train Loss: {:.7f},  G Loss: {:.7f}'.format(epoch_loss, epoch_loss_G))
        # print('Total training time is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print()
    # np.savetxt('result/'+args.dataset+'_tea_loss.txt', training_loss_tea_results, fmt='%.4f')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # plt.plot(training_loss_tea_results)
    # plt.title('α'+str(alpha)+'β'+str(beta)+'γ'+str(gamma))
    # plt.show()

    # 获取教师端的输出特征
    model_tea.eval()
    train_size = len(train_loader) * args.batch_size
    img_tea = torch.zeros(size=(train_size, args.hash_dim))
    text_tea = torch.zeros(size=(train_size, args.hash_dim))
    MultiLabels = torch.zeros(size=(train_size, args.hash_dim))
    i = 0
    for imgs, txts, labels in train_loader2:

        if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
            print("Data contains Nan.")
        # zero the parameter gradients

        # forward
        # track history if only in train
        with torch.no_grad():
            # Get model outputs and calculate loss
            # Special case for inception because in training it has an auxiliary output. In train
            #   mode we calculate the loss by summing the final output and the auxiliary output
            #   but in testing we only consider the final output.
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                txts = txts.cuda()
                labels = labels.cuda()
                simples_adj = gen_simples_adj(labels).cuda()
            else:
                simples_adj = gen_simples_adj(labels)

            # zero the parameter gradients
            # Forward
            view1_feature_tea, view2_feature_tea, m_label, _, _, _, _, _ = model_tea(imgs, txts, simples_adj, labels, args.eta, args.lAmbda)
            img_tea[i * args.batch_size: (i + 1) * args.batch_size, :] = view1_feature_tea
            text_tea[i * args.batch_size: (i + 1) * args.batch_size, :] = view2_feature_tea
            MultiLabels[i * args.batch_size: (i + 1) * args.batch_size, :] = m_label
            # backward + optimize only if in training phase
        i = i + 1

    imgs_dict = {'imgs': img_tea.cpu().numpy()}
    txts_dict = {'txts': text_tea.cpu().numpy()}
    mlabels_dict = {'mlabels': MultiLabels.cpu().numpy()}
    hdf5storage.savemat('result/' + args.dataset + '_imgs_m.mat', imgs_dict)
    hdf5storage.savemat('result/'+args.dataset + '_txts_m.mat', txts_dict)
    hdf5storage.savemat('result/' + args.dataset + '_mlabels.mat', mlabels_dict)
    return model_tea, img_tea, text_tea, MultiLabels


def train_student(args, img_tea, text_tea, MultiLabel):
    DATA_DIR = 'data/' + args.dataset + '/'
    data_loader_stu, input_data_par = get_loader_kd(DATA_DIR, args.batch_size, img_tea, text_tea, MultiLabel, noise=args.noise, noise_data=args.noise_data)

    model_stu = StuNN(img_input_dim=input_data_par['img_dim'], text_input_dim=input_data_par['text_dim'],
                      hash_dim=args.hash_dim).cuda()
    optimizer_stu = optim.Adam(model_stu.parameters(), lr=args.lr)

    since = time.time()

    training_loss_stu_results = []
    test_map_results_i2t = []
    test_map_results_t2i = []
    model_stu.train()

    # 函数用于冻结 BatchNorm 层的参数
    def freeze_batchnorm_layers(model):
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()  # 设置为 eval 模式
                for param in module.parameters():
                    param.requires_grad = False

    for epoch in range(args.stu_epoch):
        if epoch >= 5:
            freeze_batchnorm_layers(model_stu)
        
        running_loss = 0.0
        # Iterate over data.
        for i, (imgs, txts, labels, img_tea, text_tea, m_label) in enumerate(data_loader_stu):

            if torch.sum(imgs != imgs) > 1 or torch.sum(txts != txts) > 1:
                print("Data contains Nan.")
            # zero the parameter gradients
            optimizer_stu.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                # Special case for inception because in training it has an auxiliary output. In train
                #   mode we calculate the loss by summing the final output and the auxiliary output
                #   but in testing we only consider the final output.
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    txts = txts.cuda()
                    img_tea = img_tea.cuda()
                    text_tea = text_tea.cuda()
                    m_label = m_label.cuda()
                # zero the parameter gradients
                optimizer_stu.zero_grad()
                # Forward
                img_stu, text_stu = model_stu(imgs, txts)
                loss_q1_stu = loss_l2(torch.sign(m_label), img_stu)
                loss_q2_stu = loss_l2(torch.sign(m_label), text_stu)

                # loss_q1_stu = loss_l2(torch.sign(img_tea+text_tea), img_stu)
                # loss_q2_stu = loss_l2(torch.sign(img_tea+text_tea), text_stu)
                loss_q_stu = loss_q1_stu + loss_q2_stu
                img_tea = F.normalize(img_tea)
                text_tea = F.normalize(text_tea)
                m_label = F.normalize(m_label)
                img_stu = F.normalize(img_stu)
                text_stu = F.normalize(text_stu)
                loss_p = calc_loss_stu(img_tea, text_tea, m_label, img_stu, text_stu)
                loss = loss_p + args.delta * loss_q1_stu + args.delta * loss_q2_stu
                # backward + optimize only if in training phase
                loss.backward()
                optimizer_stu.step()
                if i + 1 == len(data_loader_stu) and (epoch + 1) % 2 == 0:
                    print(
                        'Epoch [%3d/%3d], Loss: %.4f, loss_p: %.4f, loss_q: %.4f'
                        % (epoch + 1, args.stu_epoch, loss.item(), loss_p.item(), loss_q_stu.item()*args.delta))
            # statistics
            running_loss += loss.item()
        # epoch_loss = running_loss
        # training_loss_stu_results.append(round(epoch_loss, 5))
        # time_elapsed = time.time() - since
        if (epoch+1) % 101 == 0:
            img2txt, txt2img = test(args,model_stu)

            print('Img2Txt: {:.4f}  Txt2Img: {:.4f}'.format(img2txt, txt2img))
            # test_map_results_i2t.append(round(img2txt, 4))
            # test_map_results_t2i.append(round(txt2img, 4))
        # print('Training time is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # print()
    # np.savetxt('result/' + args.dataset + '_map_i2t.txt', test_map_results_i2t, fmt='%.5f')
    # np.savetxt('result/' + args.dataset + '_map_t2i.txt', test_map_results_t2i, fmt='%.5f')
    # np.savetxt('result/'+args.dataset+'_stu_loss.txt', training_loss_stu_results, fmt='%.5f')
    # plt.plot(training_loss_stu_results)
    # plt.show()
    # plt.plot(test_map_results)
    # plt.show()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('...Training is completed...')
    return model_stu


def test(args, model_stu):
    DATA_DIR = 'data/' + args.dataset + '/'
    data_loader, input_data_par = get_loader(DATA_DIR, args.batch_size, train_shuffle=True, noise=0, noise_data=0)
    train_loader = data_loader['train']
    query_loader = data_loader['query']
    retrieval_loader = data_loader['retrieval']
    best_acc = 0.0
    test_img_acc_history = []
    test_txt_acc_history = []
    epoch_loss_history = []
    t_imgs, t_txts, t_labels = [], [], []
    r_imgs, r_txts, r_labels = [], [], []

    model_stu.eval()

    with torch.no_grad():
        # since = time.time()
        for imgs_test, txts_test, labels_test in query_loader:
            if torch.cuda.is_available():
                imgs_test = imgs_test.cuda()
                txts_test = txts_test.cuda()
                labels_test = labels_test.float().cuda()
            imgs_feature_test, txts_feature_test = model_stu(imgs_test, txts_test)
            t_imgs.append(imgs_feature_test.cpu().numpy())
            t_txts.append(txts_feature_test.cpu().numpy())
            t_labels.append(labels_test.cpu().numpy())
        # time_elapsed = time.time() - since
        # print('Encoding in {}'.format(time_elapsed))
        for imgs_retrieval, txts_retrieval, labels_retrieval in retrieval_loader:
            if torch.cuda.is_available():
                imgs_retrieval = imgs_retrieval.cuda()
                txts_retrieval = txts_retrieval.cuda()
                labels_retrieval = labels_retrieval.float().cuda()
            imgs_feature_retrieval, txts_feature_retrieval = model_stu(imgs_retrieval, txts_retrieval)
            r_imgs.append(imgs_feature_retrieval.cpu().numpy())
            r_txts.append(txts_feature_retrieval.cpu().numpy())
            r_labels.append(labels_retrieval.cpu().numpy())
    t_imgs = np.concatenate(t_imgs)
    t_txts = np.concatenate(t_txts)
    t_labels = np.concatenate(t_labels)
    t_imgs = np.sign(t_imgs)
    t_txts = np.sign(t_txts)

    r_imgs = np.concatenate(r_imgs)
    r_txts = np.concatenate(r_txts)
    r_labels = np.concatenate(r_labels)
    r_imgs = np.sign(r_imgs)
    r_txts = np.sign(r_txts)
    # imgs_dict = {'imgs': r_imgs}
    # txts_dict = {'txts': r_txts}

    # hdf5storage.savemat('result/'+args.dataset + '_imgs.mat', imgs_dict)
    # hdf5storage.savemat('result/'+args.dataset + '_txts.mat', txts_dict)
    #
    #
    img2text = fx_calc_map_label(t_imgs, r_txts, t_labels, r_labels)
    txt2img = fx_calc_map_label(t_txts, r_imgs, t_labels, r_labels)
    print('Bit length: {:.4f}   Img2Txt: {:.4f}  Txt2Img: {:.4f}  Average MAP: {:.4f}'.format(args.hash_dim, img2text, txt2img, (img2text + txt2img) / 2))
    # k = [1,50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000]
    # tok_img2text = calc_topk_precision(t_imgs, r_txts, t_labels, r_labels, k)
    # tok_text2img = calc_topk_precision(t_txts, r_imgs, t_labels, r_labels, k)
    # for i in range(21):
    #     print('K: {:.5f}  tok_img2text: {:.5f}  tok_text2img: {:.5f}'.format(k[i], tok_img2text[i], tok_text2img[i]))

    test_img_acc_history.append(img2text)
    test_txt_acc_history.append(txt2img)
    return img2text, txt2img

