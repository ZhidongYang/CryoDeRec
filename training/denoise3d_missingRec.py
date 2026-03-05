#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
import glob
import time
import datetime
import multiprocessing as mp

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from topaz.utils.data.loader import load_image
from topaz.utils.image import downsample

from utils.fourier import apply_fourier_mask_to_tomo
from utils.missing_wedge import (get_missing_wedge_mask, get_rotated_missing_wedge_mask)
from utils.rotation_tomo import rotate_vol_around_axis

from scipy import spatial
import topaz.mrc as mrc
import topaz.cuda

from topaz.denoise import UDenoiseNet3D


name = 'denoise3d'
help = 'denoise 3D volumes with various denoising algorithms'
BASE_SEED = 888

base_dir = '/data/wxs/tomo_denoise/CryoDeRec/cryoderec'


def sampling(tomo, scale, mode='nearest'):
    tomo = torch.from_numpy(tomo)
    tomo = tomo.unsqueeze(0).unsqueeze(0)
    tomo = F.interpolate(tomo, size=(int(tomo.shape[2]), int(tomo.shape[3] * scale), int(tomo.shape[4] * scale)), mode=mode)
    # tomo = F.interpolate(tomo, size=(int(tomo.shape[2] * scale), int(tomo.shape[3] * scale), int(tomo.shape[4] * scale)), mode=mode)
    tomo = tomo.squeeze(0).squeeze(0)
    tomo = np.array(tomo)
    return tomo


def gaussian_filter_3d(sigma, s=11):
    dim = s//2
    xx,yy,zz = np.meshgrid(np.arange(-dim, dim+1), np.arange(-dim, dim+1), np.arange(-dim,dim+1))
    d = xx**2 + yy**2 + zz**2
    f = np.exp(-0.5*d/sigma**2)
    return f


class GaussianDenoise3d(nn.Module):
    def __init__(self, sigma, scale=5):
        super(GaussianDenoise3d, self).__init__()
        width = 1 + 2*int(np.ceil(sigma*scale))
        f = gaussian_filter_3d(sigma, s=width)
        f /= f.sum()

        self.filter = nn.Conv3d(1, 1, width, padding=width//2)
        self.filter.weight.data[:] = torch.from_numpy(f).float()
        self.filter.bias.data.zero_()

    def forward(self, x):
        return self.filter(x)


def write_mrc(x, path):
    with open(path, 'wb') as f:
        mrc.write(f, x)


def add_arguments(parser):

    parser.add_argument('volumes', nargs='*', help='volumes to denoise')
    parser.add_argument('-o', '--output', default=os.path.join(base_dir, 'test_results'), help='directory to save denoised volumes')

    parser.add_argument('-m', '--model', default='unet-3d', help='use pretrained denoising model. accepts path to a previously saved model or one of the provided pretrained models. pretrained model options are: unet-3d, unet-3d-10a, unet-3d-20a (default: unet-3d)')

    ## training parameters
    parser.add_argument('-a', '--even-train-path', default=None, help='path to even training data')
    parser.add_argument('-b', '--odd-train-path', default=None, help='path to odd training data')

    parser.add_argument('--N-train', type=int, default=270, help='Number of train points per volume (default: 1000)')
    parser.add_argument('--N-test', type=int, default=50, help='Number of test points per volume (default: 200)')

    parser.add_argument('-c', '--crop', type=int, default=96, help='training tile size (default: 96)')
    parser.add_argument('-mw', '--mw_angle', type=int, default=60, help='mask for missing wedge (default: 50)')
    parser.add_argument('--base-kernel-width', type=int, default=11, help='width of the base convolutional filter kernel in the U-net model (default: 11)')

    parser.add_argument('--optim', choices=['adam', 'adagrad', 'sgd'], default='adagrad', help='optimizer (default: adagrad)')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate for the optimizer (default: 0.001)')
    parser.add_argument('--criteria', default='L2', choices=['L1', 'L2'], help='training criteria (default: L2)')
    parser.add_argument('--momentum', type=float, default=0.8, help='momentum parameter for SGD optimizer (default: 0.8)')
    parser.add_argument('--batch-size', type=int, default=2, help='minibatch size (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=200, help='number of training epochs (default: 500)')


    parser.add_argument('-w', '--weight_decay', type=float, default=0.01, help='L2 regularizer on the generative network (default: 0)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--save-prefix', default=os.path.join(base_dir, 'model_training/cryoderec_model'), help='path prefix to save denoising model')

    parser.add_argument('--num-workers', type=int, default=8, help='number of workers for dataloader (default: 1)')
    parser.add_argument('-j', '--num-threads', type=int, default=0, help='number of threads for pytorch, 0 uses pytorch defaults, <0 uses all cores (default: 0)')


    ## denoising parameters
    parser.add_argument('-g', '--gaussian', type=float, default=0, help='standard deviation of Gaussian filter postprocessing, 0 means no postprocessing (default: 0)')
    parser.add_argument('-s', '--patch-size', type=int, default=96, help='denoises volumes in patches of this size. not used if <1 (default: 96)')
    parser.add_argument('-p', '--patch-padding', type=int, default=32, help='padding around each patch to remove edge artifacts (default: 48)')

    ## other parameters
    parser.add_argument('-d', '--device', type=int, default=3, help='compute device/s to use (default: -2, multi gpu), set to >= 0 for single gpu, set to -1 for cpu')

    parser.add_argument('--masked-loss-weight', type=float, default=0.1, help='loss中masked项的权重 (default: 0.1)')


def train_epoch(iterator, model, cost_func, optim, epoch=1, num_epochs=1, N=1, use_cuda=False, masked_loss_weight=0.1):
    
    c = 0
    loss_accum = 0    
    model.train()

    for batch_idx , (source,masked,target), in enumerate(iterator):
        
        b = source.size(0)        
        loss_mb = 0
        if use_cuda:
            source = source.cuda()
            masked = masked.cuda()
            target = target.cuda()
            
        denoised_source = model(source)
        denoised_masked = model(masked)
        loss = cost_func(denoised_source, target) + masked_loss_weight * cost_func(denoised_masked, target)

        loss.backward()
        optim.step()
        optim.zero_grad()

        loss = loss.item()

        c += b
        delta = b*(loss - loss_accum)
        loss_accum += delta/c

        template = '# [{}/{}] training {:.1%}, Error={:.5f}'
        line = template.format(epoch+1, num_epochs, c/N, loss_accum)
        print(line, end='\r', file=sys.stderr)
    
    print(' '*80, end='\r', file=sys.stderr)    
    return loss_accum


def eval_model(iterator, model, cost_func, epoch=1, num_epochs=1, N=1, use_cuda=False, masked_loss_weight=0.4):
    
    c = 0
    loss_accum = 0
    model.eval()
    
    with torch.no_grad():
        for batch_idx , (source,masked,target), in enumerate(iterator):
            
            b = source.size(0)        
            loss_mb = 0
            if use_cuda:
                source = source.cuda()
                masked = masked.cuda()
                target = target.cuda()
                
            denoised_source = model(source)
            denoised_masked = model(masked)
            loss = cost_func(denoised_source, target) + masked_loss_weight * cost_func(denoised_masked, target)

            loss = loss.item()
    
            c += b
            delta = b*(loss - loss_accum)
            loss_accum += delta/c
    
            template = '# [{}/{}] testing {:.1%}, Error={:.5f}'
            line = template.format(epoch+1, num_epochs, c/N, loss_accum)
            print(line, end='\r', file=sys.stderr)
            
            
    print(' '*80, end='\r', file=sys.stderr)    
    return loss_accum


class TrainingDataset3D(torch.utils.data.Dataset):
    
    def __init__(self,even_path,odd_path,tilesize,mw_angle,N_train,N_test):

        self.tilesize = tilesize
        self.mw_angle = mw_angle
        self.N_train = N_train
        self.N_test = N_test
        self.mode = 'train'
        
        self.even_paths = []
        self.odd_paths = []

        if os.path.isfile(even_path) and os.path.isfile(odd_path):
            self.even_paths.append(even_path)
            self.odd_paths.append(odd_path)
        elif os.path.isdir(even_path) and os.path.isdir(odd_path):
            for epath in glob.glob(even_path + os.sep + '*'):
                name = os.path.basename(epath)
                opath = odd_path + os.sep + name 
                if not os.path.isfile(opath):
                    print('# Error: name mismatch between even and odd directory,', name, file=sys.stderr)
                    print('# Skipping...', file=sys.stderr)
                else:
                    self.even_paths.append(epath)
                    self.odd_paths.append(opath)

        self.means = []
        self.stds = []
        self.even = []
        self.odd = []
        self.train_idxs = []
        self.test_idxs = []

        for i,(f_even,f_odd) in enumerate(zip(self.even_paths, self.odd_paths)):
            even = self.load_mrc(f_even)
            odd = self.load_mrc(f_odd)
            # even = sampling(tomo=even, scale=2.0, mode='nearest')
            # odd = sampling(tomo=odd, scale=2.0, mode='nearest')
            if even.shape != odd.shape:
                print('# Error: shape mismatch:', f_even, f_odd, file=sys.stderr)
                print('# Skipping...', file=sys.stderr)
            else:
                even_mean,even_std = self.calc_mean_std(even)
                odd_mean,odd_std = self.calc_mean_std(odd)
                self.means.append((even_mean,odd_mean))
                self.stds.append((even_std,odd_std))  

                self.even.append(even)
                self.odd.append(odd)

                mask = np.ones(even.shape, dtype=np.uint8)
                train_idxs, test_idxs = self.sample_coordinates(mask, N_train, N_test, vol_dims=(tilesize, tilesize, tilesize))

                        
                self.train_idxs += train_idxs
                self.test_idxs += test_idxs

        if len(self.even) < 1:
            print('# Error: need at least 1 file to proceeed', file=sys.stderr)
            sys.exit(2)

    def load_mrc(self, path):
        with open(path, 'rb') as f:
            content = f.read()
        tomo,_,_ = mrc.parse(content)
        tomo = tomo.astype(np.float32)
        return tomo
    
    def get_train_test_idxs(self,dim):
        assert len(dim) == 2
        t = self.tilesize
        x = np.arange(0,dim[0]-t,t,dtype=np.int32)
        y = np.arange(0,dim[1]-t,t,dtype=np.int32)
        xx,xy = np.meshgrid(x,y)
        xx = xx.reshape(-1)
        xy = xy.reshape(-1)
        lattice_pts = [list(pos) for pos in zip(xx,xy)]
        n_val = int(self.test_frac*len(lattice_pts))
        test_idx = np.random.choice(np.arange(len(lattice_pts)),
                                   size=n_val,replace=False)
        test_pts = np.hstack([lattice_pts[idx] for idx in test_idx]).reshape((-1,2))
        mask = np.ones(dim,dtype=np.int32)
        for pt in test_pts:
            mask[pt[0]:pt[0]+t-1,pt[1]:pt[1]+t-1] = 0
            mask[pt[0]-t+1:pt[0],pt[1]-t+1:pt[1]] = 0
            mask[pt[0]-t+1:pt[0],pt[1]:pt[1]+t-1] = 0
            mask[pt[0]:pt[0]+t-1,pt[1]-t+1:pt[1]] = 0
    
        mask[-t:,:] = 0
        mask[:,-t:] = 0
        
        train_pts = np.where(mask)
        train_pts = np.hstack([list(pos) for pos in zip(train_pts[0],
                                                train_pts[1])]).reshape((-1,2))
        return train_pts, test_pts
    
    def sample_coordinates(self, mask, num_train_vols, num_val_vols, vol_dims=(96, 96, 96)):
        
        #This function is borrowed from:
        #https://github.com/juglab/cryoCARE_T2T/blob/master/example/generate_train_data.py
        """
        Sample random coordinates for train and validation volumes. The train and validation 
        volumes will not overlap. The volumes are only sampled from foreground regions in the mask.
        
        Parameters
        ----------
        mask : array(int)
            Binary image indicating foreground/background regions. Volumes will only be sampled from 
            foreground regions.
        num_train_vols : int
            Number of train-volume coordinates.
        num_val_vols : int
            Number of validation-volume coordinates.
        vol_dims : tuple(int, int, int)
            Dimensionality of the extracted volumes. Default: ``(96, 96, 96)``
            
        Returns
        -------
        list(tuple(slice, slice, slice))
            Training volume coordinates.
         list(tuple(slice, slice, slice))
            Validation volume coordinates.
        """

        dims = mask.shape
        cent = (np.array(vol_dims) / 2).astype(np.int32)
        mask[:cent[0]] = 0
        mask[-cent[0]:] = 0
        mask[:, :cent[1]] = 0
        mask[:, -cent[1]:] = 0
        mask[:, :, :cent[2]] = 0
        mask[:, :, -cent[2]:] = 0
        
        tv_span = np.round(np.array(vol_dims) / 2).astype(np.int32)
        span = np.round(np.array(mask.shape) * 0.1 / 2 ).astype(np.int32)
        val_sampling_mask = mask.copy()
        val_sampling_mask[:, :span[1]] = 0
        val_sampling_mask[:, -span[1]:] = 0
        val_sampling_mask[:, :, :span[2]] = 0
        val_sampling_mask[:, :, -span[2]:] = 0

        foreground_pos = np.where(val_sampling_mask == 1)
        sample_inds = np.random.choice(len(foreground_pos[0]), 2, replace=False)
    
        val_sampling_mask = np.zeros(mask.shape, dtype=np.int8)
        val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        for z, y, x in zip(*val_sampling_inds):
            val_sampling_mask[z - span[0]:z + span[0],
            y - span[1]:y + span[1],
            x - span[2]:x + span[2]] = mask[z - span[0]:z + span[0],
                                            y - span[1]:y + span[1],
                                            x - span[2]:x + span[2]].copy()
    
            mask[max(0, z - span[0] - tv_span[0]):min(mask.shape[0], z + span[0] + tv_span[0]),
            max(0, y - span[1] - tv_span[1]):min(mask.shape[1], y + span[1] + tv_span[1]),
            max(0, x - span[2] - tv_span[2]):min(mask.shape[2], x + span[2] + tv_span[2])] = 0
    
        foreground_pos = np.where(val_sampling_mask)
        sample_inds = np.random.choice(len(foreground_pos[0]), num_val_vols, replace=num_val_vols<len(foreground_pos[0]))
        val_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        val_coords = []
        for z, y, x in zip(*val_sampling_inds):
            val_coords.append(tuple([slice(z-tv_span[0], z+tv_span[0]),
                                     slice(y-tv_span[1], y+tv_span[1]),
                                     slice(x-tv_span[2], x+tv_span[2])]))
    
        foreground_pos = np.where(mask)
        sample_inds = np.random.choice(len(foreground_pos[0]), num_train_vols, replace=num_train_vols < len(foreground_pos[0]))
        train_sampling_inds = [fg[sample_inds] for fg in foreground_pos]
        train_coords = []
        for z, y, x in zip(*train_sampling_inds):
            train_coords.append(tuple([slice(z - tv_span[0], z + tv_span[0]),
                                     slice(y - tv_span[1], y + tv_span[1]),
                                     slice(x - tv_span[2], x + tv_span[2])]))
        
        return train_coords, val_coords

    def calc_mean_std(self, tomo):
        finite_mask = np.isfinite(tomo)
        if not np.any(finite_mask):
            raise ValueError("All values in tomo are non-finite (inf/NaN).")

        vals = tomo[finite_mask].astype(np.float32)

        vals = np.clip(vals, -1e6, 1e6)

        mu = vals.mean(dtype=np.float64)
        std = vals.std(dtype=np.float64)
        if (not np.isfinite(std)) or std < 1e-6:
            std = 1.0

        return float(mu), float(std)

    # modified on 14th May by Zhidong Yang
    def _sample_rot_axis_and_angle(self, idx):
        seed = BASE_SEED
        rotvec = torch.from_numpy(
            spatial.transform.Rotation.random(random_state=seed).as_rotvec()
        )
        rot_axis = rotvec / rotvec.norm()
        rot_angle = torch.rad2deg(rotvec.norm())
        return rot_axis, rot_angle
    
    # def _sample_rot_axis_and_angle(self, idx):
    #     seed = BASE_SEED + idx  # 每个样本不同的随机数种子
    #     rng = np.random.default_rng(seed)
    #     rot = spatial.transform.Rotation.random(random_state=rng)
    #     rotvec_np = np.asarray(rot.as_rotvec(), dtype=np.float32)
    #     rotvec = torch.from_numpy(rotvec_np)
    #     rot_axis = rotvec / rotvec.norm()
    #     rot_angle = torch.rad2deg(rotvec.norm())
    #     return rot_axis, rot_angle
    
    # def _sample_rot_axis_and_angle(self, idx):
    #     seed = BASE_SEED
    #     rot = spatial.transform.Rotation.random(random_state=seed)
    #     rotvec_raw = rot.as_rotvec()
    #     rotvec_np = np.array([rotvec_raw[0], rotvec_raw[1], rotvec_raw[2]], dtype=np.float32)
    #     rotvec = torch.from_numpy(rotvec_np)
    #     rot_axis = rotvec / rotvec.norm()
    #     rot_angle = torch.rad2deg(rotvec.norm())
    #     return rot_axis, rot_angle

    def __len__(self):
        if self.mode == 'train':
            return self.N_train * len(self.even)
        else:
            return self.N_test * len(self.even)
            
    def __getitem__(self, idx):
        
        if self.mode == 'train':
            Idx = int(idx / self.N_train)
            idx = self.train_idxs[idx]
        else:
            Idx = int(idx / self.N_test)
            idx = self.test_idxs[idx]

        even = self.even[Idx]
        odd = self.odd[Idx]
       
        mean = self.means[Idx]
        std = self.stds[Idx]
        
        even_ = even[idx]
        odd_ = odd[idx]

        # modified on 14th May by Zhidong Yang
        # print(idx)
        rot_axis, rot_angle = self._sample_rot_axis_and_angle(idx)  # 随机获取一个旋转轴和角度
        even_ = rotate_vol_around_axis(
            even_,
            rot_angle=rot_angle,
            rot_axis=rot_axis
        )
        odd_ = rotate_vol_around_axis(
            odd_,
            rot_angle=rot_angle,
            rot_axis=rot_axis
        )
        mw_mask = get_missing_wedge_mask(
            grid_size=even_.shape,
            mw_angle=self.mw_angle
        )

        even_ = np.array(even_)
        odd_ = np.array(odd_)
        even_masked_ = apply_fourier_mask_to_tomo(even_, mw_mask)
        
        
        eps = 1e-6
        even_ = (even_ - mean[0]) / (std[0] + eps)
        even_masked_ = (even_masked_ - mean[0]) / (std[0] + eps)
        odd_ = (odd_ - mean[1]) / (std[1] + eps)
        even_, even_masked_, odd_ = self.augment(even_, even_masked_, odd_)

        even_ = np.expand_dims(even_, axis=0)
        even_masked_ = np.expand_dims(even_masked_, axis=0)
        odd_ = np.expand_dims(odd_, axis=0)
        
        source = torch.from_numpy(even_).float()
        masked = torch.from_numpy(even_masked_).float()
        target = torch.from_numpy(odd_).float()

        return source, masked, target

    def set_mode(self, mode):
        modes = ['train', 'test']
        assert mode in modes
        self.mode = mode 

    def augment(self, x, y, z):
        # mirror axes
        for ax in range(3):
            if np.random.rand() < 0.5:
                x = np.flip(x, axis=ax)
                y = np.flip(y, axis=ax)
                z = np.flip(z, axis=ax)

        # rotate around each axis
        for ax in [(0,1), (0,2), (1,2)]:
            k = np.random.randint(4)
            x = np.rot90(x, k=k, axes=ax)
            y = np.rot90(y, k=k, axes=ax)
            z = np.rot90(z, k=k, axes=ax)

        return np.ascontiguousarray(x), np.ascontiguousarray(y), np.ascontiguousarray(z)


def train_model(even_path, odd_path, save_prefix, save_interval, device
               , base_kernel_width=11
               , cost_func='L1'
               , weight_decay=0
               , learning_rate=0.001
               , optim='adagrad'
               , momentum=0.8
               , minibatch_size=10
               , num_epochs=500
               , N_train=1000
               , N_test=200
               , tilesize=96
               , mw_angle=50
               , num_workers=1
               , masked_loss_weight=0.4
               ):
    output = sys.stdout
    log = sys.stderr

    if save_prefix is not None:
        save_dir = os.path.dirname(save_prefix)
        if len(save_dir) > 0 and not os.path.exists(save_dir):
            print('# creating save directory:', save_dir, file=log)
            os.makedirs(save_dir)
        log_file_path = os.path.join(save_dir, "train_log.txt")
    else:
        log_file_path = "train_log.txt"

    start_time = time.time()
    now = datetime.datetime.now()
    print('# starting time: {:02d}/{:02d}/{:04d} {:02d}h:{:02d}m:{:02d}s'.format(now.month,now.day,now.year,now.hour,now.minute,now.second), file=log)

    # initialize the model
    print('# initializing model...', file=log)
    model_base = UDenoiseNet3D(base_width=7)
    model,use_cuda,num_devices = set_device(model_base, device)
    
    if cost_func == 'L2':
        cost_func = nn.MSELoss()
        print(cost_func)
    elif cost_func == 'L1':
        cost_func = nn.L1Loss()
        print(cost_func)
    else:
        cost_func = nn.MSELoss()

    wd = weight_decay
    params = [{'params': model.parameters(), 'weight_decay': wd}]
    lr = learning_rate
    if optim == 'sgd':
        optim = torch.optim.SGD(params, lr=lr, momentum=momentum)
    elif optim == 'rmsprop':
        optim = torch.optim.RMSprop(params, lr=lr)
    elif optim == 'adam':
        optim = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8, amsgrad=True)
    elif optim == 'adagrad':
        optim = torch.optim.Adagrad(params, lr=lr)
    else:
        raise Exception('Unrecognized optim: ' + optim)

    # Load the data
    print('# loading data...', file=log)
    if not (os.path.isdir(even_path) or os.path.isfile(even_path)):
        print('ERROR: Cannot find file or directory:', even_path, file=log)
        sys.exit(3)
    if not (os.path.isdir(odd_path) or os.path.isfile(odd_path)):
        print('ERROR: Cannot find directory:', odd_path, file=log)
        sys.exit(3)
    
    if tilesize < 1:
        print('ERROR: tilesize must be >0', file=log)
        sys.exit(4)
    if tilesize < 10:
        print('WARNING: small tilesize is not recommended', file=log)
    data = TrainingDataset3D(even_path, odd_path, tilesize, mw_angle, N_train, N_test)
    
    N_train = len(data)
    data.set_mode('test')
    N_test = len(data)
    data.set_mode('train')
    num_workers = min(num_workers, mp.cpu_count())
    digits = int(np.ceil(np.log10(num_epochs)))

    iterator = torch.utils.data.DataLoader(data,batch_size=minibatch_size,num_workers=num_workers,shuffle=False)
    
    ## Begin model training
    print('# training model...', file=log)
    print('\t'.join(['Epoch', 'Split', 'Error', 'Time']), file=output)

    best_loss = float('inf')  # 新增：记录最优loss

    for epoch in range(num_epochs):
        epoch_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        data.set_mode('train')
        epoch_loss_accum = train_epoch(iterator,
                                       model,
                                       cost_func,
                                       optim,
                                       epoch=epoch,
                                       num_epochs=num_epochs,
                                       N=N_train,
                                       use_cuda=use_cuda,
                                       masked_loss_weight=masked_loss_weight)
        line = '\t'.join([str(epoch+1), 'train', str(epoch_loss_accum), epoch_time])
        print(line, file=output)
        with open(log_file_path, "a") as logf:
            logf.write(line + "\n")
        
        # evaluate on the test set
        data.set_mode('test')
        epoch_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        epoch_loss_accum = eval_model(iterator,
                                   model,
                                   cost_func,
                                   epoch=epoch,
                                   num_epochs=num_epochs,
                                   N=N_test,
                                   use_cuda=use_cuda,
                                   masked_loss_weight=masked_loss_weight)
        line = '\t'.join([str(epoch+1), 'test', str(epoch_loss_accum), epoch_time])
        print(line, file=output)
        with open(log_file_path, "a") as logf:
            logf.write(line + "\n")

        ## save the models
        if save_prefix is not None and (epoch+1)%save_interval == 0:
            model.eval().cpu()
            save_model(model, epoch+1, save_prefix, digits=digits)
            if use_cuda:
                model.cuda()

        # 新增：保存best model
        if save_prefix is not None and epoch_loss_accum < best_loss:
            best_loss = epoch_loss_accum
            model.eval().cpu()
            best_path = save_prefix + '_best.sav'
            torch.save(model, best_path)
            print(f'# Best model saved at epoch {epoch+1} with loss {best_loss:.5f}', file=log)
            if use_cuda:
                model.cuda()

    print('# training completed!', file=log)

    end_time = time.time()
    now = datetime.datetime.now()
    print("# ending time: {:02d}/{:02d}/{:04d} {:02d}h:{:02d}m:{:02d}s".format(now.month,now.day,now.year,now.hour,now.minute,now.second), file=log)
    print("# total time:", time.strftime("%Hh:%Mm:%Ss", time.gmtime(end_time - start_time)), file=log)

    return model_base, num_devices


def save_model(model, epoch, save_prefix, digits=3):
    if type(model) is nn.DataParallel:
        model = model.module

    path = save_prefix + ('_epoch{:0'+str(digits)+'}.sav').format(epoch) 
    #path = save_prefix + '_epoch{}.sav'.format(epoch)
    torch.save(model, path)


def load_model(path, base_kernel_width=11):
    from collections import OrderedDict
    log = sys.stderr

    # load the model
    pretrained = False
    if path == 'unet-3d': # load the pretrained unet model
        name = 'unet-3d-10a-v0.2.4.sav'
        model = UDenoiseNet3D(base_width=7)
        pretrained = True
    elif path == 'unet-3d-10a':
        name = 'unet-3d-10a-v0.2.4.sav'
        model = UDenoiseNet3D(base_width=7)
        pretrained = True
    elif path == 'unet-3d-20a':
        name = 'unet-3d-20a-v0.2.4.sav'
        model = UDenoiseNet3D(base_width=7)
        pretrained = True
    
    if pretrained:
        print('# loading pretrained model:', name, file=log)

        import pkg_resources
        pkg = __name__
        path = '../pretrained/denoise/' + name
        f = pkg_resources.resource_stream(pkg, path)
        state_dict = torch.load(f) # load the parameters

        model.load_state_dict(state_dict)

    else:
        model = torch.load(path)
        if type(model) is OrderedDict:
            state = model
            model = UDenoiseNet3D(base_width=base_kernel_width)
            model.load_state_dict(state)
    model.eval()

    return model


def set_device(model, device, log=sys.stderr):
    # set the device or devices
    d = device
    use_cuda = (d != -1) and torch.cuda.is_available()
    num_devices = 1
    if use_cuda:
        device_count = torch.cuda.device_count()
        try:
            if d >= 0:
                assert d < device_count
                torch.cuda.set_device(d)
                print('# using CUDA device:', d, file=log)
            elif d == -2:
                print('# using all available CUDA devices:', device_count, file=log)
                num_devices = device_count
                model = nn.DataParallel(model)
            else:
                raise ValueError
        except (AssertionError, ValueError):
            print('ERROR: Invalid device id or format', file=log)
            sys.exit(1)
        except Exception:
            print('ERROR: Something went wrong with setting the compute device', file=log)
            sys.exit(2)

    if use_cuda:
        model.cuda()

    return model, use_cuda, num_devices


class PatchDataset:
    def __init__(self, tomo, patch_size=96, padding=48):
        self.tomo = tomo
        self.patch_size = patch_size
        self.padding = padding

        nz,ny,nx = tomo.shape

        pz = int(np.ceil(nz/patch_size))
        py = int(np.ceil(ny/patch_size))
        px = int(np.ceil(nx/patch_size))
        self.shape = (pz,py,px)
        self.num_patches = pz*py*px


    def __len__(self):
        return self.num_patches

    def __getitem__(self, patch):
        # patch index
        i,j,k = np.unravel_index(patch, self.shape)

        patch_size = self.patch_size
        padding = self.padding
        tomo = self.tomo

        # pixel index
        i = patch_size*i
        j = patch_size*j
        k = patch_size*k

        # make padded patch
        d = patch_size + 2*padding
        x = np.zeros((d, d, d), dtype=np.float32)

        # index in tomogram
        si = max(0, i-padding)
        ei = min(tomo.shape[0], i+patch_size+padding)
        sj = max(0, j-padding)
        ej = min(tomo.shape[1], j+patch_size+padding)
        sk = max(0, k-padding)
        ek = min(tomo.shape[2], k+patch_size+padding)

        # index in crop
        sic = padding - i + si
        eic = sic + (ei - si)
        sjc = padding - j + sj
        ejc = sjc + (ej - sj)
        skc = padding - k + sk
        ekc = skc + (ek - sk)

        x[sic:eic,sjc:ejc,skc:ekc] = tomo[si:ei,sj:ej,sk:ek]
        return np.array((i,j,k), dtype=int),x


def denoise(model, path, outdir, patch_size=128, padding=128, batch_size=1
           , volume_num=1, total_volumes=1):
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    with open(path, 'rb') as f:
        content = f.read()
    tomo,header,_ = mrc.parse(content)
    tomo = tomo.astype(np.float32)
    name = os.path.basename(path)

    # Up Sampling for Structure Enhancement
    # tomo = sampling(tomo, scale=2.0, mode='nearest')
    mu = tomo.mean()
    std = tomo.std()
    # denoise in patches
    d = next(iter(model.parameters())).device
    denoised = np.zeros_like(tomo)

    with torch.no_grad():
        if patch_size < 1:
            x = (tomo - mu)/std
            x = torch.from_numpy(x).to(d)
            x = model(x.unsqueeze(0).unsqueeze(0)).squeeze().cpu().numpy()
            x = std*x + mu
            denoised[:] = x
        else:
            patch_data = PatchDataset(tomo, patch_size, padding)
            total = len(patch_data)
            count = 0

            batch_iterator = torch.utils.data.DataLoader(patch_data, batch_size=batch_size)
            for index,x in batch_iterator:
                x = x.to(d)
                x = (x - mu)/std
                x = x.unsqueeze(1) # batch x channel

                # denoise
                x = model(x)
                x = x.squeeze(1).cpu().numpy()

                # stitch into denoised volume
                for b in range(len(x)):
                    i,j,k = index[b]
                    xb = x[b]

                    patch = denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size]
                    pz,py,px = patch.shape

                    xb = xb[padding:padding+pz,padding:padding+py,padding:padding+px]
                    denoised[i:i+patch_size,j:j+patch_size,k:k+patch_size] = xb

                    count += 1
                    print('# [{}/{}] {:.2%}'.format(volume_num, total_volumes, count/total), name, file=sys.stderr, end='\r')

            print(' '*100, file=sys.stderr, end='\r')


    ## save the denoised tomogram
    outpath = outdir + os.sep + name

    # Down Sampling to original size
    # denoised = sampling(denoised, scale=0.5, mode='nearest')

    # use the read header except for a few fields
    header = header._replace(mode=2) # 32-bit real
    header = header._replace(amin=denoised.min())
    header = header._replace(amax=denoised.max())
    header = header._replace(amean=denoised.mean())

    with open(outpath, 'wb') as f:
        mrc.write(f, denoised, header=header)


def main(args):
    # set the number of threads
    num_threads = args.num_threads
    from topaz.torch import set_num_threads
    set_num_threads(num_threads)

    # do denoising
    model = None
    do_train = (args.even_train_path is not None) or (args.odd_train_path is not None)
    if do_train:
        print('# training denoising model!', file=sys.stderr)
        model, num_devices = train_model(args.even_train_path, args.odd_train_path
                           , args.save_prefix, args.save_interval
                           , args.device
                           , base_kernel_width=args.base_kernel_width
                           , cost_func=args.criteria
                           , learning_rate=args.lr
                           , optim=args.optim
                           , momentum=args.momentum
                           , minibatch_size=args.batch_size
                           , num_epochs=args.num_epochs
                           , N_train=args.N_train
                           , N_test=args.N_test
                           , tilesize=args.crop
                           , num_workers=args.num_workers
                           , masked_loss_weight=args.masked_loss_weight
                           )

    if len(args.volumes) > 0: # tomograms to denoise!
        if model is None: # need to load model
            model = load_model(args.model, base_kernel_width=args.base_kernel_width)

        gaussian_sigma = args.gaussian
        if gaussian_sigma > 0:
            print('# apply Gaussian filter postprocessing with sigma={}'.format(gaussian_sigma), file=sys.stderr)
            model = nn.Sequential(model, GaussianDenoise3d(gaussian_sigma))
        model.eval()
        
        model, use_cuda, num_devices = set_device(model, args.device)

        #batch_size = args.batch_size
        #batch_size *= num_devices
        batch_size = num_devices

        patch_size = args.patch_size
        padding = args.patch_padding
        print('# denoising with patch size={} and padding={}'.format(patch_size, padding), file=sys.stderr)
        # denoise the volumes
        total = len(args.volumes)
        count = 0
        for path in args.volumes:
            count += 1
            denoise(model, path, args.output
                   , patch_size=patch_size
                   , padding=padding
                   , batch_size=batch_size
                   , volume_num=count
                   , total_volumes=total
                   )



if __name__ == '__main__':
    import argparse
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(help)
    add_arguments(parser)
    args = parser.parse_args()
    main(args)