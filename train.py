'''
Train Vision Transfromer classifier
author: phatnt
date: June-14-2022
'''
import os
import os
import sys
import shutil
import argparse
import numpy as np
from datetime import date

import torch
from tqdm.autonotebook import tqdm
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from dataset import CustomDataset, Normalize, ToTensor, Resize

from vit_pytorch import SimpleViT

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='path to train/test/val.txt contain folder')
    parser.add_argument('--imgsz', type=int, default=256, help='Train image size')
    parser.add_argument('--channels',type=int, default=3)
    parser.add_argument('--patch', type=int, default=16)
 
    parser.add_argument('--resume', type=str, default=None, help='Path to resume weight for resume previous-training')
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--work-dir', type=str, default=None)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    return parser.parse_args()

def save_checkpoint(state, saved_path, is_best_loss):
    '''
        Save training model in a specific folder.
        Args:
            state: torch state dict.
            saved_path: model saved place.
            is_best_loss: is this model reach lowest loss?
    '''
    assert state is not None, '[ERROR]: state dict is none!'
    os.makedirs(saved_path, exist_ok=True)
    saved_file = os.path.join(saved_path, 'last.pth') 
    torch.save(state, saved_file)
    if is_best_loss:
        shutil.copyfile(saved_file, saved_file.replace('last', 'best_loss'))
    print('[INFO] Checkpoint saved!')

def adjust_learning_rate(optimizer, epoch_num, init_lr):
    '''
        Sets the learning rate to the initial LR decayed by 10 every 30 epochs.
        Args:
            optimizer: training optimizer.
            epoch_num: current epoch num.
            learning_rate: init learning rate
    '''
    lr = init_lr * (0.1 ** (epoch_num // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(args):
    '''
        Train efficient net.
    '''
    today = date.today()
    assert args.batch > 0, '[ERROR] Batch size must > 0'
    assert args.epoch > 0, '[ERROR] Max epoch must > 0'
    assert os.path.isdir(args.data), f'[ERROR] Could not found {args.data}. Or not a directory!'
    
    # Load model
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(args.device)
    else:
        device = torch.device("cpu")

    if args.resume:
        assert os.path.isfile(args.resume), f'[ERROR] Could not found {args.resume}.'
        print(f"[INFO] Loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location=device)
        num_classes = checkpoint['num_classes']
        in_channels = checkpoint['in_channels']
        imgsz = checkpoint['image_size']
        patch_size = checkpoint['patch_size']
        last_epoch = checkpoint['epoch']
    else:
        in_channels = args.channels
        imgsz = args.imgsz
        patch_size = args.patch
        last_epoch = 0

    num_classes = 1
    model = SimpleViT(
        image_size = imgsz,
        channels = in_channels,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )

    if args.resume:
        model.load_state_dict(checkpoint['state_dict'])
        print(f"[INFO] Loaded checkpoint {args.resume}. At epoch {last_epoch}.")

    # Load data 
    train_file = os.path.join(args.data, "train.txt")
    assert os.path.isfile(train_file), f'[ERROR] Could not found train.txt in {args.data}'
    
    train_transforms = transforms.Compose([Resize(imgsz), Normalize(), ToTensor()])
    train_loader = torch.utils.data.DataLoader(CustomDataset(train_file, train_transforms),
                                                batch_size=args.batch, shuffle=True,
                                                num_workers=args.workers, pin_memory=False, sampler=None)
    
    print('[INFO] Model info:')
    print(f'\t + Using SimpVit.')
    print(f'\t + Input image size: {imgsz} x {imgsz} x {args.channels}.')
    print(f'\t + Output Classes: {num_classes}.')

    model.to(device=device, dtype=torch.float)
    torch.backends.cudnn.benchmark = True

    # Loss & Optimizer
    criterion = torch.nn.MSELoss().to(device=device, dtype=torch.float)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    if args.resume:
        optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    # Create log and weight saved folder
    saved_folder_name = f'{today.strftime("%b-%d")}_{os.path.basename(args.data)}' if args.work_dir is None else args.work_dir
    log_dir = os.path.join('./logs/', saved_folder_name)
    weight_save_dir = os.path.join('./result/', saved_folder_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weight_save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f'[INFO] Log Dirs: {log_dir} Created!')
    print(f'[INFO] Weights Dirs: {weight_save_dir} Created')

    init_lr = args.lr
    max_epoch = args.epoch
    best_loss = 1e5
    try:
        for epoch in range(1, max_epoch+1):
            if epoch < last_epoch:
                continue
            model.train()
            epoch_loss = []
            is_best_loss = False
            adjust_learning_rate(optimizer, epoch, init_lr)
            progress_bar = tqdm(total=len(train_loader))

            for i, sample_batched in enumerate(train_loader):
                images = sample_batched['image']
                targets = sample_batched['target']
                images = images.to(device=device, dtype=torch.float)
                targets = targets.to(device=device, dtype=torch.float)
                optimizer.zero_grad()
                output = torch.squeeze(model(images))
                loss = criterion(output, targets)
                loss.backward()
                optimizer.step()
                epoch_loss.append(float(loss))
                progress_bar.set_description( f"Epoch: {epoch}/{max_epoch}. Loss: {loss:.5f}.")
                progress_bar.update(1)
            scheduler.step(np.mean(epoch_loss))
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Training Loss', loss, epoch * len(train_loader) + i)
            writer.add_scalar('Training Learning rate', current_lr, epoch)
            
            if (loss <= best_loss):
                best_loss = loss
                is_best_loss = True
            writer.close()
            progress_bar.close()
            print(f"[INFO] Epoch {epoch}: Loss: {loss:.5f}.")
            save_checkpoint({
                'epoch': epoch,
                'image_size': imgsz,
                'in_channels': in_channels,
                'num_classes': num_classes,
                'patch_size': patch_size,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'num_classes': num_classes},
                saved_path=weight_save_dir, is_best_loss=is_best_loss)
            last_epoch = epoch
    except KeyboardInterrupt:
        writer.close()
        progress_bar.close()
        save_checkpoint({
            'epoch': last_epoch,
            'image_size': imgsz,
            'in_channels': in_channels,
            'num_classes': num_classes,
            'patch_size': patch_size,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict()},
            saved_path=weight_save_dir, is_best_loss=False)
    
if __name__ == '__main__':
    args = parser_args()
    train(args)
