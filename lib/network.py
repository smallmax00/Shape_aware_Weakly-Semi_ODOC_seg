import torch
import torch.nn as nn
import torch.nn.functional as F
from .Res2Net_v1b import res2net50_v1b_26w_4s



class region_edge_detector(nn.Module):
    def __init__(self, max_k=9, min_k=9):
        super(region_edge_detector, self).__init__()
        self.max_pooling_1 = nn.MaxPool2d(min_k, 1, min_k//2)


    def forward(self, x):
        eros = -self.max_pooling_1(-x)
        edge = x - eros
        return edge


class sdm_edge_detector(nn.Module):
    def __init__(self, max_k=9, min_k=9):
        super(sdm_edge_detector, self).__init__()
        self.max_pooling_1 = nn.MaxPool2d(min_k, 1, min_k//2)



    def forward(self, x):
        x = 2 * torch.sigmoid(torch.relu(x) * 5000) - 1
        eros = -self.max_pooling_1(-x)
        edge = x - eros
        return edge


class SDM_region_detector(nn.Module):
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2
        super(SDM_region_detector, self).__init__()

    def forward(self, x):

        region = 2 * torch.sigmoid(torch.relu(x) * 5000) - 1

        return region


class Cal_vcdr(nn.Module):
    def __init__(self, pi=3.14):
        self.pi = pi
        super(Cal_vcdr, self).__init__()
    def forward(self, region, boundary, smooth = 1e-6):
        """
        Perimeter = Pi * sqrt (2 * (a**2 + b**2) )
        Area = Pi * a * b
        """
        b_cup = torch.sum(boundary[:, 0, ...], dim=(1, 2)) / 9
        r_cup = torch.sum(region[:, 0, ...], dim=(1, 2))
        b_disc =torch.sum(boundary[:, 1, ...], dim=(1, 2)) / 9
        r_disc = torch.sum(region[:, 1, ...], dim=(1, 2))
        """
        Euler method 
        """
        a1 = (torch.pow(b_cup, 2) + torch.sqrt((4 * self.pi * r_cup + b_cup**2) * torch.abs(b_cup ** 2 - 4 * self.pi * r_cup))) / 4 * pow(self.pi, 2)

        a2 = (torch.pow(b_disc, 2) + torch.sqrt((4 * self.pi * r_disc + b_disc**2) * torch.abs(b_disc ** 2 - 4 * self.pi * r_disc))) / 4 * pow(self.pi, 2)

        est_vcdr = torch.sqrt((a1 + smooth) / (a2 + smooth))



        return est_vcdr


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MSRF(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSRF, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class graph_node_r(nn.Module):
    def __init__(self, channel, out_channel):
        super(graph_node_r, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)

        self.conv5 = nn.Conv2d(4 * channel, out_channel, 1)

    def forward(self, x1, x2, x3):
        up_x1 = self.conv_upsample1(self.upsample(x1))
        conv_x2 = self.conv_1(x2)
        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))

        up_x2 = self.conv_upsample2(self.upsample(x2))
        conv_x3 = self.conv_2(x3)
        cat_x3 = self.conv_concat3(torch.cat((up_x2, conv_x3), 1))

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2))
        conv_cat_x3 = self.conv_3(cat_x3)
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, conv_cat_x3), 1))
        x = self.conv5(cat_x4)
        return x


class graph_node_s(nn.Module):
    def __init__(self, channel, out_channel):
        super(graph_node_s, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat4 = BasicConv2d(4 * channel, 4 * channel, 3, padding=1)

        self.conv5 = nn.Conv2d(4 * channel, out_channel, 1)

    def forward(self, x1, x2, x3):
        up_x1 = self.conv_upsample1(self.upsample(x1))
        conv_x2 = self.conv_1(x2)
        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))

        up_x2 = self.conv_upsample2(self.upsample(x2))
        conv_x3 = self.conv_2(x3)
        cat_x3 = self.conv_concat3(torch.cat((up_x2, conv_x3), 1))

        up_cat_x2 = self.conv_upsample3(self.upsample(cat_x2))
        conv_cat_x3 = self.conv_3(cat_x3)
        cat_x4 = self.conv_concat4(torch.cat((up_cat_x2, conv_cat_x3), 1))
        x = self.conv5(cat_x4)
        return x


class ODOC_cdr_graph(nn.Module):

    def __init__(self, channel=64, k1=5000, k2=70, dropout=0.1):
        super(ODOC_cdr_graph, self).__init__()
        # self.k = k
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)

        self.rfb2_1 = MSRF(256, channel)
        self.rfb3_1 = MSRF(512, channel)
        self.rfb4_1 = MSRF(1024, channel)
        self.rfb5_1 = MSRF(2048, channel)

        self.graph_node_r = graph_node_r(channel=channel, out_channel=2)
        self.graph_node_s = graph_node_s(channel=channel, out_channel=2)


        self.r_edge = region_edge_detector(min_k=9)
        self.s_edge = sdm_edge_detector(min_k=9)

        self.r_edge_cdr = region_edge_detector(min_k=3)
        self.s_edge_cdr = sdm_edge_detector(min_k=3)

        self.s_region = SDM_region_detector(k1, k2)
        self.dropout = nn.Dropout2d(p=dropout, inplace=False)
        self.cal_vcdr = Cal_vcdr(pi=3.14)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)

        x2 = self.resnet.layer2(x1)

        x3 = self.resnet.layer3(x2)

        x4 = self.resnet.layer4(x3)

        x2_o = self.rfb3_1(x2)

        x3_o = self.rfb4_1(x3)

        x4_o = self.rfb5_1(x4)

        x_all_r = self.graph_node_r(x4_o, x3_o, x2_o)
        x_all_r = F.interpolate(x_all_r, scale_factor=2, mode='bilinear', align_corners=True)

        x_all_s = self.graph_node_s(x4_o, x3_o, x2_o)
        x_all_s = F.interpolate(x_all_s, scale_factor=2, mode='bilinear', align_corners=True)

        if self.training:
            x_all_r = self.dropout(x_all_r)
        r_gcn = x_all_r
        if self.training:
            x_all_s = self.dropout(x_all_s)
        s_gcn = x_all_s
        r_gcn = F.interpolate(r_gcn, scale_factor=4, mode='bilinear')
        s_gcn = F.interpolate(s_gcn, scale_factor=4, mode='bilinear')

        outputs_region = torch.sigmoid(r_gcn)
        outputs_sdm = torch.tanh_(s_gcn)

        # detect edge from region
        est_edge_r = self.r_edge(outputs_region)
        # detect edge and convert region from signed distance map
        est_region_s = self.s_region(outputs_sdm)
        # detect edge from sdm
        est_edge_s = self.s_edge(outputs_sdm)
        # detect contour
        est_edge_r_cdr = self.r_edge_cdr(outputs_region)

        est_edge_r_cdr = F.interpolate(est_edge_r_cdr, scale_factor=4, mode='bilinear')


        outputs_est_boundary = est_edge_r
        outputs_est_region_sdm = est_region_s
        outputs_est_boundary_sdm = est_edge_s

        # cal vcdr
        outputs_est_cdr_region = self.cal_vcdr(outputs_region, est_edge_r_cdr)

        return outputs_region, outputs_sdm, outputs_est_boundary,outputs_est_region_sdm, outputs_est_boundary_sdm, outputs_est_cdr_region


