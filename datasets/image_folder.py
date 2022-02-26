import os
import json
from PIL import Image

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

        if split_file is None:
            # os.listdir => 返回指令路徑下的檔案名稱
            # filenames = ['0001.png', '0002.png',......]
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if first_k is not None:
            filenames = filenames[:first_k]


        self.files = []
        for filename in filenames:
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

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        # 應該可以針對這個去做修改，maybe random sample degree, 我們把不同scene中相同degree的放在一起
        # 每次 getitem 就先sample degree, 不同drgree就對應到不同的 file list, 裡面去裝那個degree 的blur img 路徑
        # 以 cache = none 為例
        # 這裡取出idx 對應的img 絕對位置
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            # 取出真實 img 影像
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            return x


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
