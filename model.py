import math
import torch.nn.functional as F
import torch
from torch import nn
from torch.autograd import Function
from torch.nn import Parameter
from torchvision import models

from util import normalize, set_to_zero, gen_normalize

criterion = nn.CrossEntropyLoss()


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class HAGCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self,num_classes):
        super(HAGCN, self).__init__()
        self.weight = Parameter(torch.rand(num_classes, num_classes))

    def forward(self, input, adj):
        adj = adj * self.weight
        output = torch.matmul(adj, input)
        return output


class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = models.vgg19_bn(pretrained=True)
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        return features


class TxtMLP(nn.Module):
    def __init__(self, code_len=300, txt_bow_len=1386, num_class=24):
        super(TxtMLP, self).__init__()
        self.fc1 = nn.Linear(txt_bow_len, 4096)
        self.fc2 = nn.Linear(4096, code_len)
        self.classifier = nn.Linear(code_len, num_class)

    def forward(self, x):
        feat = F.leaky_relu(self.fc1(x), 0.2)
        feat = F.leaky_relu(self.fc2(feat), 0.2)
        predict = self.classifier(feat)
        return feat, predict


class ImgNN(nn.Module):
    """Network to learn image representations"""

    def __init__(self, input_dim=4096, output_dim=1024):
        super(ImgNN, self).__init__()
        self.img_nn = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.img_nn(x)
        return out


class TextNN(nn.Module):
    """Network to learn text representations"""

    def __init__(self, input_dim=1024, output_dim=1024):
        super(TextNN, self).__init__()
        self.text_nn = nn.Sequential(
            nn.Linear(input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.text_nn(x)
        return out


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Discriminator(nn.Module):

    def __init__(self, batch_size):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(batch_size*2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = ReverseLayerF.apply(x, 1)
        return self.discriminator(x)


class Classifier(nn.Module):

    def __init__(self, hash_dim, num_classes):
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hash_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)



class TeaNN(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, num_classes=10,
                 inp=None, in_channel=300, batch_size=100, hash_dim=128, dropout=0.):
        super(TeaNN, self).__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.img_net = ImgNN(img_input_dim, minus_one_dim)
        self.text_net = TextNN(text_input_dim, minus_one_dim)

        self.discriminator1 = Discriminator(batch_size)
        self.discriminator2 = Discriminator(batch_size)
        self.classifier_simple = Classifier(hash_dim,num_classes)

        if inp is not None:
            self.inp = Parameter(inp, requires_grad=False)
        else:
            self.inp = Parameter(torch.rand(num_classes, in_channel))
        self.fc_label = nn.Sequential(
            nn.Linear(in_channel, minus_one_dim),
            nn.Tanh(),
            nn.Linear(minus_one_dim, minus_one_dim),
            nn.Tanh()
        )

        # self.gc1_label = HAGCN(num_classes)
        # self.gc2_label = HAGCN(num_classes)
        self.gc1_label = GraphConvolution(self.inp.size()[1], minus_one_dim, dropout)
        self.gc2_label = GraphConvolution(minus_one_dim, minus_one_dim, dropout)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.weight_adj_label1 = Parameter(torch.rand(num_classes, num_classes))
        # self.bias_adj_label1 = Parameter(0.02*torch.rand(num_classes, num_classes))
        self.weight_adj_label2 = Parameter(torch.rand(num_classes, num_classes))
        # self.bias_adj_label2 = Parameter(0.02*torch.rand(num_classes, num_classes))
        self.weight_m = Parameter(torch.rand(num_classes, hash_dim))

        self.classifier = nn.Sequential(
            nn.Linear(minus_one_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        self.fc_se = nn.Sequential(
            nn.Linear(batch_size*2, batch_size*2),
            # nn.Linear(batch_size*4, batch_size*2)
        )
        self.fc_fe = nn.Sequential(
            nn.Linear(batch_size * 2, batch_size * 2),
            # nn.Linear(batch_size * 4, batch_size * 2)
        )
        self.gc1_simple = GraphConvolution(minus_one_dim, minus_one_dim, dropout)
        self.gc2_simple = GraphConvolution(minus_one_dim, minus_one_dim, dropout)

        self.hash_img = nn.Sequential(
            nn.Linear(minus_one_dim, hash_dim),
            nn.Tanh()
        )
        self.hash_txt = nn.Sequential(
            nn.Linear(minus_one_dim, hash_dim),
            nn.Tanh()
        )
        self.hash_label = nn.Sequential(
            nn.Linear(minus_one_dim, hash_dim),
            nn.Tanh()
        )

    def forward(self, feature_img, feature_text, adj_se, y, eta, lAmbda):
        view1_feature = self.img_net(feature_img)
        view2_feature = self.text_net(feature_text)

        # 获取label特征
        # label = self.fc_label(self.inp)
        adj_label = F.cosine_similarity(self.inp.unsqueeze(1), self.inp.unsqueeze(0), dim=-1)
        adj_label = set_to_zero(adj_label, 0.4)
        label1 = self.gc1_label(self.inp, gen_normalize(adj_label*self.weight_adj_label1))
        label2 = self.gc2_label(label1, gen_normalize(adj_label*self.weight_adj_label2))

        # 各层的分类损失
        matrix_i = torch.eye(self.inp.size()[0]).float().cuda()
        l2_loss = nn.MSELoss()
        # label_predict = self.classifier(se)
        label_predict1 = self.classifier(label1)
        label_predict2 = self.classifier(label2)
        # loss_cla0 = (1/self.num_classes)*l2_loss(matrix_i, label_predict)
        # loss_cla1 = (1/self.num_classes)*l2_loss(matrix_i, label_predict1)
        # loss_cla2 = (1/self.num_classes)*l2_loss(matrix_i, label_predict2)
        # loss_cla0 = criterion(label_predict,matrix_i)
        loss_cla1 = criterion(label_predict1, matrix_i)
        loss_cla2 = criterion(label_predict2, matrix_i)
        # loss_cla_triplet10 = F.relu(loss_cla1 - loss_cla0 + 0.01)
        loss_cla_triplet21 = F.relu(loss_cla2 - loss_cla1 + 0.01)

        loss_cla = (1/2)*(loss_cla1 + loss_cla2)
        loss_cla_triplet = loss_cla_triplet21

        # 由标签获取多标签
        # m_label = self.tanh(torch.matmul(y.float(), label2 * self.weight_m))
        # m_label = self.hash_label(m_label)
        label2 = self.hash_label(label2)
        m_label = self.tanh(torch.matmul(y.float(), label2 * self.weight_m))

        # 合并邻接矩阵 n*n(adj_se) ==> 2n*2n(adj_se)
        adj_se1 = torch.cat([adj_se, adj_se], dim=0)
        adj_se = torch.cat([adj_se1, adj_se1], dim=1)

        adj_fe11 = F.cosine_similarity(view1_feature.unsqueeze(1), view1_feature.unsqueeze(0), dim=-1)
        adj_fe12 = F.cosine_similarity(view1_feature.unsqueeze(1), view2_feature.unsqueeze(0), dim=-1)
        adj_fe22 = F.cosine_similarity(view2_feature.unsqueeze(1), view2_feature.unsqueeze(0), dim=-1)

        # 合并邻接矩阵 n*n ==> 2n*2n(adj_fe)
        fe_block_1 = torch.cat([adj_fe11, adj_fe12], dim=1)
        fe_block_2 = torch.cat([adj_fe12.t(), adj_fe22], dim=1)
        adj_fe = torch.cat([fe_block_1, fe_block_2], dim=0)

        # 将值小于eta的元素设置为零
        adj_fe = set_to_zero(adj_fe, eta)

        adj_se_w = self.fc_se(adj_se)
        adj_fe_w = self.fc_fe(adj_fe)
        # adj_sf = adj_se + adj_fe_w
        # adj_fs = adj_fe + adj_se_w
        adj_sf = adj_se + adj_fe_w
        adj_fs = adj_fe + adj_se_w

        adj_a = lAmbda*adj_sf + (1-lAmbda)*adj_fs
        # adj_a = 0.5*adj_fe + 0.5*adj_se
        adj_a = normalize(adj_a, self.batch_size*2)
        simple_feature = torch.cat([view1_feature, view2_feature], dim=0)
        x = self.gc1_simple(simple_feature, adj_a)
        x = self.relu(x)
        x = self.gc2_simple(x, adj_a)
        x = self.relu(x)

        n = x.size(0)
        imgs = x[:n//2, :]
        texts = x[n//2:, :]
        imgs = self.hash_img(imgs)
        texts = self.hash_txt(texts)

        norm_img = torch.norm(imgs, dim=1)[:, None] * torch.norm(label2, dim=1)[None, :] + 1e-6
        norm_txt = torch.norm(texts, dim=1)[:, None] * torch.norm(label2, dim=1)[None, :] + 1e-6
        norm_m_label = torch.norm(m_label, dim=1)[:, None] * torch.norm(label2, dim=1)[None, :] + 1e-6
        label2 = label2.transpose(0, 1)
        y_img = torch.matmul(imgs, label2)
        y_text = torch.matmul(texts, label2)
        y_m_label = torch.matmul(m_label, label2)
        y_img = y_img / norm_img
        y_text = y_text / norm_txt
        y_m_label = y_m_label / norm_m_label
        loss_simple_cla = (((y_img - y.float()) ** 2).sum(1).sqrt().mean() + ((y_text - y.float()) ** 2).sum(1).sqrt().mean()
                            + ((y_m_label - y.float()) ** 2).sum(1).sqrt().mean())
        # return imgs, texts, m_label, loss_adj, loss_cla, loss_mcla
        return imgs, texts, m_label, loss_cla, loss_cla_triplet, loss_simple_cla, adj_sf, adj_fs

    def get_config_optim(self, lr):
        param_groups = []
        for name, param in self.named_parameters():
            if 'bias_adj_label1' in name or 'bias_adj_label2' in name or 'weight_adj_label1' in name or 'weight_adj_label2' in name:
                param_groups.append({'params': param, 'lr': lr})
            else:
                # 其他层的学习率设置为 0.001
                param_groups.append({'params': param, 'lr': lr})
        return param_groups


class StuImageNet(nn.Module):
    def __init__(self, img_input_dim=4096,  minus_one_dim=1024):
        super(StuImageNet, self).__init__()
        self.img_nn = nn.Sequential(
            nn.Linear(img_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, minus_one_dim),
            nn.BatchNorm1d(minus_one_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.img_nn(x)
        return out


class StuTextNet(nn.Module):
    def __init__(self, text_input_dim=1024,  minus_one_dim=1024):
        super(StuTextNet, self).__init__()
        self.img_nn = nn.Sequential(
            nn.Linear(text_input_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.Tanh(),
            nn.Linear(2048, minus_one_dim),
            nn.BatchNorm1d(minus_one_dim),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.img_nn(x)
        return out


class StuNN(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=1024, minus_one_dim=1024, hash_dim=128):
        super(StuNN, self).__init__()
        self.imgNN_stu = StuImageNet(img_input_dim, minus_one_dim)
        self.textNN_stu = StuTextNet(text_input_dim, minus_one_dim)
        self.hash_img = nn.Sequential(
            nn.Linear(minus_one_dim, hash_dim),
            nn.BatchNorm1d(hash_dim),
            nn.Tanh()
        )
        self.hash_txt = nn.Sequential(
            nn.Linear(minus_one_dim, hash_dim),
            nn.BatchNorm1d(hash_dim),
            nn.Tanh()
        )

    def forward(self, feature_img, feature_text):
        view1_feature = self.imgNN_stu(feature_img)
        view1_feature = self.hash_img(view1_feature)

        view2_feature = self.textNN_stu(feature_text)
        view2_feature = self.hash_txt(view2_feature)
        return view1_feature, view2_feature
