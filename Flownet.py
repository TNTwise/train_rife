import torch
import torch.nn as nn
import torch.nn.functional as F
from warplayer import warp
import numpy as np
# from refine import *

device = torch.device("cuda")

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.LeakyReLU(0.2, True)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, True)
    )

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, feat=False):
        x0 = self.cnn0(x)
        x = self.relu(x0)
        x1 = self.cnn1(x)
        x = self.relu(x1)
        x2 = self.cnn2(x)
        x = self.relu(x2)
        x3 = self.cnn3(x)
        if feat:
            return [x0, x1, x2, x3]
        return x3                        
    
class ResConv(nn.Module):
    def __init__(self, c, dilation=1):
        super(ResConv, self).__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, dilation, dilation=dilation, groups=1)
        self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
        self.relu = nn.LeakyReLU(0.2, True)
        
    def forward(self, x):
        return self.relu(self.conv(x) * self.beta + x)

'''
class ResBlock(nn.Module):
    def __init__(self, planes):
        super(ResBlock, self).__init__()
        self.conv1 = conv(planes, planes, 3, 1, 1)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.prelu = nn.PReLU(planes)
        self.fc1 = nn.Conv2d(planes, 16, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(16, planes, kernel_size=1, bias=True)
        
        
    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.conv2(x)
        w = x.mean(3, True).mean(2, True)
        w = self.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))        
        return self.prelu(x * w + y)
'''

class Flownet(nn.Module):
    def __init__(self, in_planes, c=64):
        super(Flownet, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock = nn.Sequential(
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
            ResConv(c),
        )
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(c, 4*13, 4, 2, 1),
            nn.PixelShuffle(2)
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False)
        if flow is not None:
            flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False) * 1. / scale
            x = torch.cat((x, flow), 1)
        feat = self.conv0(x)
        feat = self.convblock(feat)
        tmp = self.lastconv(feat)
        tmp = F.interpolate(tmp, scale_factor=scale, mode="bilinear", align_corners=False)
        flow = tmp[:, :4] * scale
        mask = tmp[:, 4:5]
        conf = tmp[:, 5:]
        return flow, mask, conf
        
class FlownetCas(nn.Module):
    def __init__(self):
        super(FlownetCas, self).__init__()
        self.block0 = Flownet(7+8, c=192)
        self.block1 = Flownet(8+4+8+8, c=128)
        self.block2 = Flownet(8+4+8+8, c=96)
        self.block3 = Flownet(8+4+8+8, c=64)
        self.block4 = Flownet(8+4+8+8, c=32)
        self.encode = Head()
        self.teacher = Flownet(8+4+8+3+8, c=64)
        self.caltime = nn.Sequential(
            nn.Conv2d(8+9, 32, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x, timestep=0.5, scale=[8, 4, 2, 1], training=False, distill=True):
        img0 = x[:, :3]
        img1 = x[:, 3:6]
        f0 = self.encode(img0)
        f1 = self.encode(img1)
        
        if not torch.is_tensor(timestep):
            timestep = (x[:, :1].clone() * 0 + 1) * timestep
        else:
            timestep = timestep.repeat(1, 1, img0.shape[2], img0.shape[3])
        timestep_label = timestep.clone()
        gt = x[:, 6:]
        if gt.shape[1] == 3:
            pred_time = self.caltime(torch.cat((x, f0, f1), 1)).mean(3, True).mean(2, True)
            if np.random.uniform(0, 1) < 0.3:
                timestep = pred_time.repeat(1, 1, img0.shape[2], img0.shape[3])
        flow_list = []
        merged = []
        flow_list_teacher = []
        mask_list = []
        feat_list = []
        teacher_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = None 
        loss_cons = 0
        stu = [self.block0, self.block1, self.block2, self.block3, self.block4]
        for i in range(5):
            if flow is not None:
                flow_d, mask, feat = stu[i](torch.cat((warped_img0, warped_img1, warped_f0, warped_f1, timestep, mask, feat), 1), flow, scale=scale[i])
                flow = flow + flow_d
            else:
                flow, mask, feat = stu[i](torch.cat((img0, img1, f0, f1, timestep), 1), None, scale=scale[i])
            mask_list.append(mask)
            flow_list.append(flow)
            feat_list.append(feat[:, :1])
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            warped_f0 = warp(f0, flow[:, :2])
            warped_f1 = warp(f1, flow[:, 2:4])
            merged_student = (warped_img0, warped_img1)
            merged.append(merged_student)
        conf = torch.clamp(torch.cat(feat_list, 1), 0.1, 1)
        conf = conf / (conf.sum(1, True) + 1e-3)
        if gt.shape[1] == 3:
            flow_teacher = 0
            mask_teacher = 0
            for i in range(5):
                flow_teacher += conf[:, i:i+1] * flow_list[i]
                mask_teacher += conf[:, i:i+1] * mask_list[i]
            warped_img0_teacher = warp(img0, flow_teacher[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher[:, 2:4])
            mask_teacher = torch.sigmoid(mask_teacher)
            merged_teacher = warped_img0_teacher * mask_teacher + warped_img1_teacher * (1 - mask_teacher)
            flow_list_teacher.append(flow_teacher)
            
            # 2 teacher
            flow_d, mask_teacher2, _ = self.teacher(torch.cat((warped_img0, warped_img1, gt, warped_f0, warped_f1, timestep, mask, feat), 1), flow, scale=2)
            flow_teacher2 = flow + flow_d
            mask_teacher2 = torch.sigmoid(mask_teacher2)
            warped_img0_teacher = warp(img0, flow_teacher2[:, :2])
            warped_img1_teacher = warp(img1, flow_teacher2[:, 2:4])
            merged_teacher2 = warped_img0_teacher * mask_teacher2 + warped_img1_teacher * (1 - mask_teacher2)

            teacher_list.append(merged_teacher)
            teacher_list.append(merged_teacher2)
            
        for i in range(5):
            mask_list[i] = torch.sigmoid(mask_list[i])
            merged[i] = merged[i][0] * mask_list[i] + merged[i][1] * (1 - mask_list[i])
            if gt.shape[1] == 3:
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher - gt).abs().mean(1, True)).float().detach()
                loss_cons += (((flow_teacher.detach() - flow_list[i]) ** 2).sum(1, True) ** 0.5 * loss_mask).mean() * 0.005
                loss_mask = ((merged[i] - gt).abs().mean(1, True) > (merged_teacher2 - gt).abs().mean(1, True)).float().detach()
                loss_cons += (((flow_teacher2.detach() - flow_list[i]) ** 2).sum(1, True) ** 0.5 * loss_mask).mean() * 0.005

        loss_time = loss_cons * 0
        if gt.shape[1] == 3:
            timestep_label = timestep_label.mean(3, True).mean(2, True)
            loss_time = (((pred_time - timestep_label) ** 2) * (timestep_label != 0.5).float()).mean()
        return flow_list, mask_list[4], merged, [teacher_list, flow_list_teacher], loss_cons, loss_time
