import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from datasets import register
from utils import to_pixel_samples


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
        # 會return 一個dict


def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, Image.BICUBIC)(
            transforms.ToPILImage()(img)))


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None):
        self.dataset = dataset # class ImageFolder object
        self.inp_size = inp_size # 48
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        # 這裡就會參考到 class ImageFolder的 repeat 去計算出來的 len
        return len(self.dataset)

    def __getitem__(self, idx):
        # dataset[idx] => 就會觸發 class ImageFolder 定義的 __getitem__(), 去取出對應的一張img本人
        # [REBLUR] 應該要在這裡return 相對應的 low degree + high degree img 還有它們的degree
        #img = self.dataset[idx]
        img_sor, img_tar, d1, d2 = self.dataset[idx]

        deg_diff = d2 - d1

        """
        # 所以這整個故事中，是LR size固定
        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        """


        # input size = 48 (LR size)
        w = self.inp_size # 48 (LR size)
        #w_hr = round(w_lr * s) # 80 (HR size)
        x0 = random.randint(0, img_sor.shape[-2] - w)
        y0 = random.randint(0, img_sor.shape[-1] - w)

        #[REBLUR] 兩張不同deg img都crop 48x48
        crop_sor = img_sor[:, x0: x0 + w, y0: y0 + w]
        crop_tar = img_tar[:, x0: x0 + w, y0: y0 + w]

        #crop_lr = resize_fn(crop_hr, w_lr)
        # resize_fn (wrapper.py裡的func), 用雙線性差值直接把你指定的image downscale到你指定的size (hr -> lr)

        if self.augment:
            hflip = random.random() < 0.5 # True
            vflip = random.random() < 0.5 # False
            dflip = random.random() < 0.5 # True
            # augment == Ture => 隨機決定要不要做上面這些augmentation 操作
            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x
            # [REBLUR] source, target img 都做augment
            crop_sor = augment(crop_sor)
            crop_tar = augment(crop_tar)

        # to_pixel_samples => 生出拉平的座標矩陣以及對應的vector
        # hr_coord: (6400, 2) (座標矩陣拉平),  hr_rgb: (6400, 3) (真實HR img 拉平)

        sor_coord, sor_rgb = to_pixel_samples(crop_sor.contiguous())
        tar_coord, tar_rgb = to_pixel_samples(crop_tar.contiguous())
        # [REBLUR] sor_coord: (2304 x 3), sor_rgb: (2304 x 2)

        """
        # ex: sample_q = 2304
        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(sor_coord), self.sample_q, replace=False)
            sor_coord = sor_coord[sample_lst]
            sor_rgb = sor_rgb[sample_lst]
            # sor_coord, sor_rgb 現在都是隨機亂數從3600 pixel corrdinate, vector中取 2304個出來的結果
        """
        """
        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]
        """
        #[REBLUR] (2304, 1), 除此之外還要用這個去算小的far_feat size 讓 crop_sor去 downscale生成
        deg_diff = torch.full((sor_rgb.shape[0], 1), deg_diff, dtype=float).float()



        return {
            #'inp': crop_lr,     #從 切出來的HR(3x 80 x 80)　直接Downscale 來的 LR img 本人 (3 x 48 x 48)
            #'coord': hr_coord,  #隨機亂數從3600 HR pixel corrdinate 取 2304個出來的座標們(不按順序)
            #'gt': hr_rgb        #隨機亂數從3600 HR pixel vector 取 2304個出來的座標們(不按順序)
            'sor_coord': sor_coord, #(2304 ,2)
            'crop_sor' : crop_sor,  #(3, 48, 48)
            'tar_rgb': tar_rgb,     #(2304 ,3)
            'deg_diff': deg_diff
        }




@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
