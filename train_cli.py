#!/usr/bin/env python3
"""
CLI Training Script for RIFE (Real-Time Intermediate Flow Estimation)
Version 4.25

This script allows training RIFE from the command line with configurable parameters.

Usage Examples:
    # Single GPU training
    python train_cli.py --data_path /path/to/data --batch_size 8 --epochs 100

    # Multi-GPU distributed training
    torchrun --nproc_per_node=4 train_cli.py --data_path /path/to/data --distributed

    # Resume from checkpoint
    python train_cli.py --data_path /path/to/data --resume /path/to/checkpoint

    # Training with custom learning rate
    python train_cli.py --data_path /path/to/data --lr 1e-4 --lr_min 1e-7
"""

import os
import cv2
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Disable OpenCV threading to avoid conflicts
cv2.setNumThreads(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train RIFE model for video frame interpolation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation data directory (defaults to data_path)')
    parser.add_argument('--data_format', type=str, default='triplet',
                        choices=['triplet', 'sequence', 'vimeo'],
                        help='Data format: triplet (img0, gt, img1), sequence (multiple frames), vimeo')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Maximum learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-7,
                        help='Minimum learning rate')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Number of warmup steps')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay for optimizer')
    
    # Model arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model weights')
    
    # Output arguments
    parser.add_argument('--exp_name', type=str, default='rife_experiment',
                        help='Experiment name for logging')
    parser.add_argument('--log_dir', type=str, default='./train_log',
                        help='Directory for saving logs and checkpoints')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--log_interval', type=int, default=200,
                        help='Log training metrics every N steps')
    parser.add_argument('--img_log_interval', type=int, default=1000,
                        help='Log images to tensorboard every N steps')
    
    # Distributed training arguments
    parser.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training (set by torchrun)')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Number of distributed processes')
    
    # Augmentation arguments
    parser.add_argument('--crop_size', type=int, default=384,
                        help='Random crop size for training')
    parser.add_argument('--no_flip', action='store_true',
                        help='Disable horizontal/vertical flipping')
    parser.add_argument('--no_reverse', action='store_true',
                        help='Disable temporal reversal augmentation')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=124,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_lpips', action='store_true',
                        help='Use LPIPS metric for validation (requires lpips package)')
    parser.add_argument('--fp16', action='store_true',
                        help='Use mixed precision training')
    
    args = parser.parse_args()
    
    # Set validation path to data path if not specified
    if args.val_data_path is None:
        args.val_data_path = args.data_path
    
    return args


class SimpleDataset(Dataset):
    """
    Simple dataset that loads image triplets (frame0, gt, frame1) from a directory.
    
    Expected directory structure for 'triplet' format:
        data_path/
            sequence_001/
                frame0.png (or .jpg)
                frame1.png (ground truth middle frame)
                frame2.png
            sequence_002/
                ...
    
    Expected directory structure for 'sequence' format:
        data_path/
            sequence_001/
                0000.png
                0001.png
                0002.png
                ...
            sequence_002/
                ...
    """
    
    def __init__(self, data_path, mode='train', crop_size=384, 
                 flip=True, reverse=True, data_format='triplet'):
        self.data_path = Path(data_path)
        self.mode = mode
        self.crop_size = crop_size
        self.flip = flip and mode == 'train'
        self.reverse = reverse and mode == 'train'
        self.data_format = data_format
        self.samples = []
        
        self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {mode}")
    
    def _load_samples(self):
        """Load sample paths from the data directory."""
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        # Check if data_path contains subdirectories (sequences)
        subdirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        
        if subdirs:
            # Each subdirectory is a sequence
            for seq_dir in sorted(subdirs):
                self._load_sequence(seq_dir)
        else:
            # All images are in one directory - try to find triplets
            self._load_flat_directory()
    
    def _load_sequence(self, seq_dir):
        """Load samples from a sequence directory."""
        # Get all image files
        img_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        images = sorted([f for f in seq_dir.iterdir() 
                        if f.suffix.lower() in img_extensions])
        
        if len(images) < 3:
            return
        
        if self.data_format == 'triplet':
            # Expect exactly 3 images: frame0, gt, frame2
            if len(images) == 3:
                self.samples.append((str(images[0]), str(images[1]), str(images[2]), 0.5))
        elif self.data_format == 'sequence':
            # Create triplets from sequence with varying timesteps
            for i in range(len(images) - 2):
                for j in range(i + 2, min(i + 8, len(images))):
                    for k in range(i + 1, j):
                        timestep = (k - i) / (j - i)
                        self.samples.append((str(images[i]), str(images[k]), 
                                           str(images[j]), timestep))
        elif self.data_format == 'vimeo':
            # Vimeo format: 7 frames per sequence
            if len(images) >= 7:
                # Use frames 1, 4, 7 (indices 0, 3, 6) as default
                self.samples.append((str(images[0]), str(images[3]), str(images[6]), 0.5))
    
    def _load_flat_directory(self):
        """Load from a flat directory with numbered images."""
        img_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        images = sorted([f for f in self.data_path.iterdir() 
                        if f.suffix.lower() in img_extensions])
        
        if len(images) >= 3:
            for i in range(0, len(images) - 2, 3):
                self.samples.append((str(images[i]), str(images[i+1]), 
                                   str(images[i+2]), 0.5))
    
    def __len__(self):
        return len(self.samples)
    
    def _read_image(self, path):
        """Read an image from disk."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        return img
    
    def _augment(self, img0, gt, img1):
        """Apply data augmentation."""
        h, w, _ = img0.shape
        
        # Random crop
        if self.mode == 'train' and h > self.crop_size and w > self.crop_size:
            x = random.randint(0, h - self.crop_size)
            y = random.randint(0, w - self.crop_size)
            img0 = img0[x:x+self.crop_size, y:y+self.crop_size]
            gt = gt[x:x+self.crop_size, y:y+self.crop_size]
            img1 = img1[x:x+self.crop_size, y:y+self.crop_size]
        
        return img0, gt, img1
    
    def __getitem__(self, index):
        img0_path, gt_path, img1_path, timestep = self.samples[index]
        
        # Load images
        img0 = self._read_image(img0_path)
        gt = self._read_image(gt_path)
        img1 = self._read_image(img1_path)
        
        # Resize if needed to ensure dimensions are compatible
        h, w = img0.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            scale = max(self.crop_size / h, self.crop_size / w) * 1.1
            new_h, new_w = int(h * scale), int(w * scale)
            img0 = cv2.resize(img0, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            gt = cv2.resize(gt, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            img1 = cv2.resize(img1, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Apply augmentation
        img0, gt, img1 = self._augment(img0, gt, img1)
        
        # Random flips
        if self.flip:
            if random.random() < 0.5:
                img0 = img0[::-1].copy()
                gt = gt[::-1].copy()
                img1 = img1[::-1].copy()
            if random.random() < 0.5:
                img0 = img0[:, ::-1].copy()
                gt = gt[:, ::-1].copy()
                img1 = img1[:, ::-1].copy()
        
        # Temporal reversal
        if self.reverse and random.random() < 0.5:
            img0, img1 = img1, img0
            timestep = 1 - timestep
        
        # Convert to tensors
        timestep = torch.tensor(timestep).reshape(1, 1, 1).float()
        img0 = torch.from_numpy(img0.copy()).permute(2, 0, 1).float()
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).float()
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).float()
        
        return torch.cat((img0, img1, gt), 0), timestep


def get_learning_rate(step, args, total_steps):
    """Cosine learning rate schedule with warmup."""
    if step < args.warmup_steps:
        mul = step / args.warmup_steps
    else:
        mul = np.cos((step - args.warmup_steps) / (total_steps - args.warmup_steps) * math.pi) * 0.5 + 0.5
    return args.lr * mul + args.lr_min


def flow2rgb(flow_map_np):
    """Convert optical flow to RGB visualization."""
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max() + 1e-6)
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)


def setup_distributed(args):
    """Setup distributed training environment."""
    if args.distributed:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            args.local_rank = int(os.environ['LOCAL_RANK'])
            args.world_size = int(os.environ['WORLD_SIZE'])
            args.rank = int(os.environ['RANK'])
        else:
            args.rank = args.local_rank
            
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        print(f"Initialized distributed training: rank {args.rank}/{args.world_size}")
    else:
        args.rank = 0
        args.local_rank = 0
        args.world_size = 1
    
    return args


def setup_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def train(args):
    """Main training function."""
    # Setup
    args = setup_distributed(args)
    setup_seeds(args.seed + args.rank)
    
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = args.rank == 0
    
    # Import model components
    from model import Model
    
    # Setup logging
    log_path = os.path.join(args.log_dir, args.exp_name)
    if is_main:
        os.makedirs(log_path, exist_ok=True)
        writer = SummaryWriter(os.path.join(log_path, 'train'))
        writer_val = SummaryWriter(os.path.join(log_path, 'validate'))
        print(f"Logging to: {log_path}")
    
    # Create datasets
    train_dataset = SimpleDataset(
        args.data_path,
        mode='train',
        crop_size=args.crop_size,
        flip=not args.no_flip,
        reverse=not args.no_reverse,
        data_format=args.data_format
    )
    
    val_dataset = SimpleDataset(
        args.val_data_path,
        mode='val',
        crop_size=args.crop_size,
        flip=False,
        reverse=False,
        data_format=args.data_format
    )
    
    # Create data loaders
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            sampler=train_sampler
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    
    if is_main:
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total steps: {total_steps}")
    
    # Create model
    model = Model(local_rank=args.local_rank if args.distributed else -1)
    
    # Load checkpoint or pretrained weights
    if args.resume:
        if is_main:
            print(f"Resuming from checkpoint: {args.resume}")
        model.load_model(args.resume, rank=args.local_rank if args.distributed else -1)
    elif args.pretrained:
        if is_main:
            print(f"Loading pretrained weights: {args.pretrained}")
        model.load_model(args.pretrained, rank=args.local_rank if args.distributed else -1)
    
    # LPIPS metric (optional)
    loss_fn_alex = None
    if args.use_lpips:
        try:
            import lpips
            loss_fn_alex = lpips.LPIPS(net='alex').to(device)
            if is_main:
                print("LPIPS metric enabled")
        except ImportError:
            if is_main:
                print("Warning: lpips package not found, LPIPS metric disabled")
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    step = 0
    best_psnr = 0
    
    if is_main:
        print("Starting training...")
    
    for epoch in range(args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0
        time_stamp = time.time()
        
        for i, (data_gpu, timestep) in enumerate(train_loader):
            data_time = time.time() - time_stamp
            time_stamp = time.time()
            
            # Move data to device
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            
            # Extract images
            imgs = data_gpu[:, :6]  # img0 and img1
            gt = data_gpu[:, 6:9]   # ground truth
            
            # Horizontal flip augmentation (batch-level)
            imgs = torch.cat((imgs, imgs.flip(-1)), 0)
            gt = torch.cat((gt, gt.flip(-1)), 0)
            timestep = torch.cat((timestep, timestep.flip(-1)), 0)
            
            # Get learning rate
            learning_rate = get_learning_rate(step, args, total_steps)
            
            # Forward and backward pass
            if args.fp16:
                with torch.cuda.amp.autocast():
                    pred, info = model.update(imgs, gt, learning_rate, training=True, 
                                            distill=True, timestep=timestep)
            else:
                pred, info = model.update(imgs, gt, learning_rate, training=True, 
                                        distill=True, timestep=timestep)
            
            train_time = time.time() - time_stamp
            time_stamp = time.time()
            epoch_loss += info['loss_l1']
            
            # Logging
            if step % args.log_interval == 0 and is_main:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/cons', info['loss_cons'], step)
                writer.add_scalar('loss/time', info['loss_time'], step)
                writer.add_scalar('loss/encode', info['loss_encode'], step)
                writer.add_scalar('loss/vgg', info['loss_vgg'], step)
                writer.add_scalar('loss/gram', info['loss_gram'], step)
            
            # Image logging
            if step % args.img_log_interval == 0 and is_main:
                gt_vis = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred_vis = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                
                for j in range(min(2, gt_vis.shape[0])):
                    imgs_concat = np.concatenate((merged_img[j], pred_vis[j], gt_vis[j]), 1)[:, :, ::-1]
                    writer.add_image(f'{j}/img', imgs_concat, step, dataformats='HWC')
                    writer.add_image(f'{j}/flow', np.concatenate((flow2rgb(flow0[j]), flow2rgb(flow1[j])), 1), 
                                   step, dataformats='HWC')
                writer.flush()
            
            # Print progress
            if is_main and i % 50 == 0:
                print(f'Epoch {epoch} [{i}/{steps_per_epoch}] '
                      f'time: {data_time:.2f}+{train_time:.2f} '
                      f'lr: {learning_rate:.2e} '
                      f'loss_l1: {info["loss_l1"]:.4e}')
            
            step += 1
        
        # End of epoch
        avg_loss = epoch_loss / steps_per_epoch
        if is_main:
            print(f'Epoch {epoch} completed. Average loss: {avg_loss:.4e}')
        
        # Validation
        if (epoch + 1) % args.eval_interval == 0:
            psnr = evaluate(model, val_loader, device, is_main, 
                          writer_val if is_main else None, step, loss_fn_alex)
            if is_main and psnr > best_psnr:
                best_psnr = psnr
                model.save_model(os.path.join(log_path, 'best'), rank=0)
                print(f'New best PSNR: {best_psnr:.2f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            model.save_model(log_path, rank=args.rank)
            if args.distributed:
                dist.barrier()
    
    if is_main:
        print(f"Training completed. Best PSNR: {best_psnr:.2f}")
        writer.close()
        writer_val.close()


def evaluate(model, val_loader, device, is_main, writer_val, step, loss_fn_alex=None):
    """Evaluate the model on validation set."""
    model.eval()
    psnr_list = []
    lpips_list = []
    
    with torch.no_grad():
        for i, (data_gpu, timestep) in enumerate(val_loader):
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            
            pred, info = model.update(imgs, gt, training=False)
            
            # Calculate PSNR
            for j in range(gt.shape[0]):
                mse = torch.mean((gt[j] - pred[j]) ** 2).cpu().item()
                psnr = -10 * math.log10(mse + 1e-8)
                psnr_list.append(psnr)
                
                if loss_fn_alex is not None and is_main:
                    lpips_val = loss_fn_alex(gt[j:j+1] * 2 - 1, pred[j:j+1] * 2 - 1).item()
                    lpips_list.append(lpips_val)
            
            # Log images for first batch
            if i == 0 and is_main and writer_val is not None:
                gt_vis = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                pred_vis = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
                
                for j in range(min(4, gt_vis.shape[0])):
                    imgs_concat = np.concatenate((merged_img[j], pred_vis[j], gt_vis[j]), 1)[:, :, ::-1]
                    writer_val.add_image(f'{j}/img', imgs_concat.copy(), step, dataformats='HWC')
    
    avg_psnr = np.mean(psnr_list)
    
    if is_main:
        print(f'Validation PSNR: {avg_psnr:.2f}')
        if writer_val is not None:
            writer_val.add_scalar('benchmark/psnr', avg_psnr, step)
            if lpips_list:
                writer_val.add_scalar('benchmark/lpips', np.mean(lpips_list), step)
    
    model.train()
    return avg_psnr


if __name__ == "__main__":
    args = parse_args()
    train(args)
