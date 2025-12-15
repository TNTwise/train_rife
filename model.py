import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
import torch.optim as optim
import itertools
from warplayer import warp
from torchstat import stat
from torch.nn.parallel import DistributedDataParallel as DDP
from Flownet import *
import torch.nn.functional as F
from loss import *
from vgg import *
from ssim import SSIM

device = torch.device("cuda")

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
        
def hard_update(target, source):
    for m1, m2 in zip(target.modules(), source.modules()):
        m1._buffers = m2._buffers.copy()
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Model:
    def __init__(self, local_rank=-1):
        self.flownet_update = FlownetCas()
        self.flownet = FlownetCas()
        self.optimG = AdamW(self.flownet_update.parameters(), lr=1e-6, weight_decay=1e-2)
        self.ss = SSIM()
        self.vgg = VGGPerceptualLoss().to(device)
        self.encode_target = Head()
        self.local_rank = local_rank
        self.device()
        if local_rank != -1:
            self.flownet_update = DDP(self.flownet_update, device_ids=[local_rank], output_device=local_rank)
            self.encode_target = DDP(self.encode_target, device_ids=[local_rank], output_device=local_rank)
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank)
        hard_update(self.encode_target, self._get_module(self.flownet_update).encode)
        hard_update(self.flownet, self.flownet_update)
    
    def _get_module(self, model):
        """Get the underlying module, handling both DDP and non-DDP cases."""
        return model.module if hasattr(model, 'module') else model

    def train(self):
        self.flownet_update.train()

    def eval(self):
        self.flownet_update.eval()

    def device(self):
        self.encode_target.to(device)
        self.flownet_update.to(device)
        self.flownet.to(device)

    def load_model(self, path, rank=0):
        def convert(param):
            if rank == -1:
                return {
                    k.replace("module.", ""): v
                    for k, v in param.items()
                    if "module." in k
                }
            else:
                return param
        if rank <= 0:
            if torch.cuda.is_available():
                param = convert(torch.load('{}/flownet.pkl'.format(path)))
                self.flownet_update.load_state_dict(param, False)
        hard_update(self.flownet, self.flownet_update)
        hard_update(self.encode_target, self._get_module(self.flownet).encode)
        
    def save_model(self, path, rank=0):
        if rank == 0:
            torch.save(self.flownet.state_dict(),'{}/flownet.pkl'.format(path))

    def encode_loss(self, X, Y):
        loss = 0
        X = self.encode_target(X, True)
        Y = self.encode_target(Y, True)
        for i in range(4):
            loss += (X[i] - Y[i].detach()).abs().mean()
        return loss
            
        
    def update(self, imgs, gt, learning_rate=0, mul=1, training=True, distill=False, timestep=0.5):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        img0 = imgs[:, :3]
        img1 = imgs[:, 3:]
        if training:
            self.train()
        else:
            self.eval()
        scale = [16, 8, 4, 2, 1]
        p = np.random.uniform(0, 1)
        if training:
            if p < 0.3:
                scale = [8, 4, 2, 2, 1]
            elif p < 0.6:
                scale = [4, 4, 2, 2, 1]
        flow, mask, merged, teacher_res, loss_cons, loss_time = self.flownet_update(torch.cat((imgs, gt), 1), timestep, scale=scale, training=training, distill=distill)
        loss_l1 = 0
        for i in range(5):
            loss_l1 *= 0.8
            loss_l1 += (merged[i] - gt).abs().mean()
        loss_l1 *= 0.1
        loss_tea = ((teacher_res[0][0] - gt).abs().mean() + (teacher_res[0][1] - gt).abs().mean()) * 0.1
        loss_cons += ((flow[-1] ** 2 + 1e-6).sum(1) ** 0.5).mean() * 1e-5
        loss_encode = 0
        for i in range(5):
            loss_encode *= 0.8
            loss_encode += self.encode_loss(merged[i], gt)
        loss_encode += self.encode_loss(teacher_res[0][0], gt) + self.encode_loss(teacher_res[0][1], gt)
        loss_encode *= 0.1
        loss_vgg, loss_gram = self.vgg(merged[-1], gt)
        if training:
            self.optimG.zero_grad()
            loss_G = (loss_vgg + loss_encode + loss_gram) + loss_tea + loss_cons + loss_l1 - self.ss(merged[-1], gt) * 0.1 + loss_time
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(self.flownet_update.parameters(), 1.0)
            self.optimG.step()
            soft_update(self.encode_target, self._get_module(self.flownet_update).encode, 0.001)
            soft_update(self.flownet, self.flownet_update, 0.001)
            flow_teacher = teacher_res[1][0]
        else:
            flow_teacher = flow[-1]
        return merged[-1], {
            'merged_tea': teacher_res[0][0],
            'mask': mask,
            'mask_tea': mask,
            'flow': flow[3][:, :2],
            'flow_tea': flow_teacher,
            'loss_l1': loss_l1,
            'loss_tea': loss_tea,
            'loss_encode': loss_encode,
            'loss_vgg': loss_vgg,
            'loss_gram': loss_gram,
            'loss_cons': loss_cons,
            'loss_time': loss_time
            }
