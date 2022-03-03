import os
import json
from PIL import Image
import random
import pickle
import imageio
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# 從 datasets.py 中把register function拿來用
from datasets import register


@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        #[REBLUR] 先讓範圍小一點
        self.deg_list = [3,7,11,15,19]
        self.scene_path_folder = os.listdir(root_path)
        self.root_path = root_path

        if self.root_path == "/home/skchen/GOPRO_final/train/":
            self.gt_num = 24

        if self.root_path == "/home/skchen/GOPRO_final/val/":
            self.gt_num = 5


        if split_file is None:
            #[REBLUR] 0.png, 1.png, ... 23.png
            filename_basename = sorted(os.listdir(root_path+ self.scene_path_folder[0] +"/gt/"))
        """
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]
        """
        #[REBLUR] 原本是 ../DIV2k/0001.png 這個方便讀取的路徑入self.files, 但因為我們現在要讓他random degree去取，就到後面在處理
        self.files = filename_basename
        """
        for filename in filename_basename:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                #files內存的是img的絕對路徑 (還沒真讀)
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                # 直接開讀，真實img資料進到file
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))
        """
    def __len__(self):

        # return len(self.files) * self.repeat
        # [REBLUR] Train 每個scene 24張，共22個scene, 每個scene又有5個degree => 2640
        # [REBLUR] Val 每個scene 5張，共22個scene, 每個scene又有5個degree => 550
        return self.gt_num * 22 * 5 * self.repeat

    def __getitem__(self, idx):
        #[REBLUR] files => gt basename, 這裡先random 兩個 degree

        scene = self.scene_path_folder[random.randint(0, len(self.scene_path_folder)-1)]

        d1 = 0
        d2 = 0
        while(d1 == d2):
            d1 = self.deg_list[random.randint(0, len(self.deg_list)-1)]
            d2 = self.deg_list[random.randint(0, len(self.deg_list)-1)]

        if d1 > d2:
            temp = d1
            d1 = d2
            d2 = temp

        idx = idx % self.gt_num

        img1_path = self.root_path + scene + "/deg_" + str(d1) + "/" + self.files[idx]
        img2_path = self.root_path + scene + "/deg_" + str(d2) + "/" + self.files[idx]

        """
        if self.gt_num == 24 :
            print("training")
            print("img1: ", img1_path)
            print("img2: ", img2_path)
        if self.gt_num == 5:
            print("val")
            print("img1: ", img1_path)
            print("img2: ", img2_path)
        """


        #x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            # 取出真實 img 影像
            #return transforms.ToTensor()(Image.open(x).convert('RGB'))
            img1 = transforms.ToTensor()(Image.open(img1_path).convert('RGB'))
            img2 = transforms.ToTensor()(Image.open(img2_path).convert('RGB'))
            return img1, img2, d1, d2

        """   
        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x
        """


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
