import argparse
import json
import os

import torch

from generate_noise_data import generate_noise_M
from model import StuNN
from trian_model import train_teacher, train_student, test


parser = argparse.ArgumentParser()
parser.add_argument('--tea_epoch', type=int, default=100, help='mirflickr--40/NUS-WIDE--60/MS-COCO--100')
parser.add_argument('--stu_epoch', type=int, default=100)

parser.add_argument('--hash_dim', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.00005, help='0.0001')
parser.add_argument('--dropout', type=float, default=0.5, help='0.5')
parser.add_argument('--dataset', type=str, default='MS-COCO', help='mirflickr/NUS-WIDE/MS-COCO')
parser.add_argument('--batch_size', type=int, default=1024, help='mirflickr--512/NUS-WIDE--1024/MS-COCO--1024')
parser.add_argument('--embedding', type=str, default='glove', help='glove/fasttext/googlenews')
parser.add_argument('--noise', type=float, default='0', help='0 false, else ture')
parser.add_argument('--noise_data', type=float, default='0.8', help='0 false, else ture')

parser.add_argument('--alpha', type=float, default=0.1, help='[Teacher]The factor of kS-cos(B, B) from DJSRH loss.')
parser.add_argument('--beta', type=float, default=0.1, help='0.1')
parser.add_argument('--gamma', type=float, default=1, help='[Teacher]The factor of SIGN loss.')
parser.add_argument('--mu', type=float, default=0.001, help='[Student]The factor of SIGN loss.')
parser.add_argument('--delta', type=float, default=0.01, help='[Student]The factor of SIGN loss.')

parser.add_argument('--eta', type=float, default=0.4, help='[Student]The factor of supervise infomation from teacher network.')
parser.add_argument('--tau', type=float, default=1, help='[Student]The factor of supervise infomation from teacher network.')
parser.add_argument('--lAmbda', type=float, default=0.5, help='[Student]The factor of supervise infomation from teacher network.')


args = parser.parse_args()


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data parameters

    EVAL = False
    # list_num1 = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    list_num1 = [16, 32, 64, 128]
    list_num2 = [0.1, 0.05, 0.01, 0.005]
    list_num3 = [0.2, 0.4, 0.6, 0.8]
    list_num4 = [0.0001, 0.00005, 0.00001, 0.000005]
    datasets = ['mirflickr', 'MS-COCO', 'NUS-WIDE']
    training_results = []
    # for dataset in datasets:
    #     args.dataset = dataset
    #     if dataset == 'mirflickr':
    #         args.tea_epoch = 10
    #         args.stu_epoch = 10
    #         args.batch_size = 512
    #     elif dataset == 'MS-COCO':
    #         args.tea_epoch = 10
    #         args.stu_epoch = 10
    #         args.batch_size = 1024
    #     elif dataset == 'NUS-WIDE':
    #         args.tea_epoch = 10
    #         args.stu_epoch = 10
    #         args.batch_size = 1024
    # for alpha in list_num2:
    #     for beta in list_num2:
    #         for gamma in list_num2:
    #             for mu in list_num2:
    #
    #                 args.alpha = alpha
    #                 args.beta = beta
    #                 args.gamma = gamma
    #                 args.mu = mu
                    #     args.dropout = dropout
                    # for dataset in datasets:

                    # for lAmbda in list_num1:
                    #     # args.alpha = alpha
                    #     # args.beta = beta
                    #     # args.gamma = gamma
                    #     args.lAmbda = lAmbda
                    # for aaa in list_num2:
                    #     args.aaa = aaa
                    # seed = 37
    # for epoch in list_num3:
    #     for lr in list_num4:
    #         args.tea_epoch = epoch
    #         args.stu_epoch = epoch
    #         args.lr = lr
    # for noise_std in list_num2:
    #     # generate_noise_M([0.2, 0.4, 0.6, 0.8], noise_std)
    for noise_data in list_num3:
        args.noise_data = noise_data
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if EVAL:
            DATA_DIR = 'data/' + args.dataset + '/'
            model_stu = StuNN(512, 512, hash_dim=args.hash_dim).cuda()
            model_stu.load_state_dict(torch.load('model/ACLCH_' + args.dataset + '.pth'))
            img2txt, txt2img = test(args, model_stu)
        else:
            model_tea, img_tea, text_tea, MultiLabel = train_teacher(args)

            model_stu = train_student(args, img_tea, text_tea, MultiLabel)

            img2txt, txt2img = test(args, model_stu)

            training_results.append(
                {'dataset': args.dataset, 'noise_data': args.noise_data,
                 'img2txt MAP': round(img2txt, 5), 'txt2img MAP': round(txt2img, 5),'MAP': round((img2txt + txt2img) / 2, 5)})
            # training_results.append(
            #     {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma, 'mu': args.mu,
            #      'MAP': round((img2txt+txt2img)/2, 5)})
            # torch.save(model_stu.state_dict(), 'model/ACLCH_' + args.dataset + '.pth')
            # training_results.append(
            #     {'epoch': args.tea_epoch, 'lr': args.lr,  'MAP': round((img2txt + txt2img) / 2, 5)})
            # torch.save(model_stu.state_dict(), 'model/ACLCH_' + args.dataset + '.pth')
            # 将训练结果保存到JSON文件中
            with open('training_results.json', 'w') as f:
                json.dump(training_results, f, indent=4)
