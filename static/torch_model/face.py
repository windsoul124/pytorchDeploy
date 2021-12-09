import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

# 服务器
# torch.cuda.set_device(0)
c = 100


class Arc(nn.Module):

    def __init__(self, feature_num=512, cls_num=113):
        super(Arc, self).__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)), requires_grad=True)
        # self.func = nn.Softmax()

    def forward(self, x, s=80, m=0.5):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / s
        a = torch.acos(cosa)
        arcsoftmax = torch.exp(
            s * torch.cos(a + m)) / (torch.sum(torch.exp(s * cosa), dim=1, keepdim=True) - torch.exp(
            s * cosa) + torch.exp(s * torch.cos(a + m)))
        return arcsoftmax


class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()
        self.sub_net = nn.Sequential(
            models.mobilenet_v2(),  # 导入mobilenet_v3
        )
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),  # 222.1指的是leakRelu负半轴的倾斜角
            nn.Linear(1000, 512, bias=False),
        )
        # self.arc_softmax = Arc(512,87)  # 121是和最终的分类的数量有关，512或256或128都形

    def forward(self, x):
        y = self.sub_net(x)  # y
        # 是原本的mobilenet_v2()的输出值
        feature = self.feature_net(y)  # self.feature_net网络导数第二层

        return feature
        # self.arc_softmax(feature)  # 前向推理返回的是特征和arc_softmax分类

    # def encode(self, x):
    #
    #     return self.feature_net(self.sub_net(x))  # 返回的是倒数第二层的值


def compare(face1, face2):
    face1_norm = F.normalize(face1)  # 数据归一化，压缩到0~1之间
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)
    # print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.T)  # 矩阵运算
    return cosa  # 相似度


loss_fn = nn.NLLLoss()

if __name__ == '__main__':
    pass
