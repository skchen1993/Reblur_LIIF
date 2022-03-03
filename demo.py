#sync test123123123123
import argparse
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='4.png')
    parser.add_argument('--model', default='/home/skchen/ML_practice/reblur/save/ex1/epoch-best.pth')
    parser.add_argument('--deg_diff', default='80')
    parser.add_argument('--output', default='output_85.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # img 照他這樣讀進來，pixel: 0~1之間
    img = transforms.ToTensor()(Image.open(args.input).convert('RGB')).cuda()
    h = img.shape[1]
    w = img.shape[2]

    #生出LIIF model, 並且把預訓練參數也load進去
    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()

    #[REBLUR] = (921600, 2) => 後面做unsqueeze(0)後變 (1, 921600, 2)
    #h, w = list(map(int, args.resolution.split(',')))
    coord = make_coord((h, w)).cuda().cuda()

    pred = batched_predict(model, ((img - 0.5) / 0.5).cuda().unsqueeze(0),
        coord.unsqueeze(0), args.deg_diff, bsize=921600)[0]
    # clamp: 把數值收緊於區間(0,1), 小於0則等於0, 大於1則等於1
    pred = (pred * 0.5 + 0.5).clamp(0, 1).view(h, w, 3).permute(2, 0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
