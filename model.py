import torch
import torch.nn as nn
import numpy as np
import os
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
        def _load_checkpoint_object(p: str):
            # Supports both directory paths (containing flownet.pkl) and direct file paths.
            if os.path.isdir(p):
                p = os.path.join(p, 'flownet.pkl')
            return torch.load(p, map_location='cpu')

        def _extract_state_dict(obj):
            # Common conventions: raw state_dict, {'state_dict': ...}, {'model': ...}, etc.
            if isinstance(obj, dict):
                if obj and all(torch.is_tensor(v) for v in obj.values()):
                    return obj
                for key in ('state_dict', 'model', 'net', 'params', 'flownet'):
                    val = obj.get(key)
                    if isinstance(val, dict) and val and all(torch.is_tensor(v) for v in val.values()):
                        return val
            raise ValueError('Unsupported checkpoint format: expected a state_dict or a dict containing one.')

        def _strip_known_prefixes(state_dict):
            # Strip DDP/DataParallel prefix.
            out = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[len('module.'):]
                # Some checkpoints save with an extra top-level prefix.
                for prefix in ('flownet_update.', 'flownet.', 'net.', 'model.'):
                    if k.startswith(prefix):
                        k = k[len(prefix):]
                out[k] = v
            return out

        if rank <= 0 and torch.cuda.is_available():
            ckpt_obj = _load_checkpoint_object(path)
            param = _strip_known_prefixes(_extract_state_dict(ckpt_obj))

            # Always load into the underlying module.
            # This makes checkpoints portable across: single-GPU, DataParallel, and DDP.
            target = self._get_module(self.flownet_update)
            # strict=False to allow backward/forward-compatible checkpoints.
            incompatible = target.load_state_dict(param, strict=False)
            missing = list(getattr(incompatible, 'missing_keys', []))
            unexpected = list(getattr(incompatible, 'unexpected_keys', []))
            if missing or unexpected:
                print(
                    f'[Model.load_model] Loaded with strict=False. '
                    f'missing={len(missing)} unexpected={len(unexpected)}'
                )
                if missing:
                    print('[Model.load_model] Missing keys (first 20):', missing[:20])
                if unexpected:
                    print('[Model.load_model] Unexpected keys (first 20):', unexpected[:20])
        hard_update(self.flownet, self.flownet_update)
        hard_update(self.encode_target, self._get_module(self.flownet).encode)
        
    def save_model(self, path, rank=0):
        if rank == 0:
            # Save with a consistent "module." prefix for compatibility with common
            # RIFE checkpoints/loaders that expect keys like "module.encode.*".
            state = self._get_module(self.flownet).state_dict()
            state = {('module.' + k) if not k.startswith('module.') else k: v for k, v in state.items()}
            torch.save(state, '{}/flownet.pkl'.format(path))

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
