import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from torch.autograd import Variable
import skimage.measure as skim
import scipy.misc
from skimage.color import rgb2yuv, yuv2rgb
from yuv_frame_io import YUV_Read,YUV_Write
import lpips
loss_fn_alex = lpips.LPIPS(net='alex').cuda()

torch.distributed.init_process_group(backend="nccl", world_size=8)
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
seed = 124
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True

from model import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from util import *
from torch.utils.data.distributed import DistributedSampler

exp = os.path.abspath('.').split('/')[-1]

log_path = '../../train_log1116/{}'.format(exp)
if local_rank == 0:
    writer = SummaryWriter(log_path + '/train')
    writer_val = SummaryWriter(log_path + '/validate')

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
    return 2e-4 * mul + 1e-7

def train(model):
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=8, pin_memory=True, num_workers=16)
    # evaluate(model, val_data, 0)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            distill = True
            learning_rate = get_learning_rate(step)
            imgs = torch.cat((imgs, imgs.flip(-1)), 0)
            gt = torch.cat((gt, gt.flip(-1)), 0)
            timestep = torch.cat((timestep, timestep.flip(-1)), 0)
            pred, info = model.update(imgs, gt, learning_rate, training=True, distill=distill, timestep=timestep)
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/cons', info['loss_cons'], step)
                writer.add_scalar('loss/time', info['loss_time'], step)
                writer.add_scalar('loss/encode', info['loss_encode'], step)
                writer.add_scalar('loss/vgg', info['loss_vgg'], step)
                writer.add_scalar('loss/gram', info['loss_gram'], step)
            if step % 1000 == 1 and local_rank == 0:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                for i in range(2):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step += 1
        nr_eval += 1
        if nr_eval % 10 == 0:
            evaluate(model, val_data, step)
        model.save_model(log_path, local_rank)    
        dist.barrier()

def evaluate(model, val_data, nr_eval):
    loss_l1_list = []
    loss_cons_list = []
    loss_tea_list = []
    psnr_list = []
    lpips_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_cons_list.append(info['loss_cons'].cpu().numpy())
        for j in range(gt.shape[0]):
            if local_rank == 0:
                lpips = loss_fn_alex(gt[j] * 2 - 1, pred[j] * 2 - 1).detach().cpu().data
                lpips_list.append(lpips)
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
            if local_rank == 0:
                lpips_list.append(lpips)
        gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
        flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
        flow1 = info['flow_tea'].permute(0, 2, 3, 1).cpu().numpy()
        if i == 0 and local_rank == 0:
            for j in range(4):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    writer_val.add_scalar('benchmark/psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('benchmark/psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)
    writer_val.add_scalar('benchmark/lpips', np.array(lpips_list).mean(), nr_eval)
        
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=150, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help='minibatch size')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    args = parser.parse_args()
    # args.step_per_epoch = 51313 // args.batch_size    
    model = Model(args.local_rank)
    train(model)
        
