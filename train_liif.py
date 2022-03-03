"""
sync test 0227
Train for generating LIIF, from image to implicit representation.
    sync_test# 0226
    Config:
        train_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        val_dataset:
          dataset: $spec; wrapper: $spec; batch_size:
        (data_norm):
            inp: {sub: []; div: []}
            gt: {sub: []; div: []}
        (eval_type):
        (eval_bsize):

        model: $spec
        optimizer: $spec
        epoch_max:
        (multi_step_lr):
            milestones: []; gamma: 0.5
        (resume): *.pth

        (epoch_val): ; (epoch_save):
"""

import argparse
import os

import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

import datasets
import models
import utils
from test import eval_psnr

import wandb


def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    # spec 就是train_dataset or val_dataset 小dict
    # 以train_dataset為例，裡面又有其他小dict => dataset, wrapper, batch_size
    # 這裡用train_dataset dict下的子dict => dataset來做 CustomDataset in pytorch
    """
    第一次的make利用了 dataset中的:
        name = 'image-folder' 指定了要用的class來建立dataset object
        args中的path去給定路徑吃img, repeat, cache方法之類的
    """
    dataset = datasets.make(spec['dataset'])
    # 上面的dataset是 class ImageFolder的object

    # 把上一部件好的CustomDataset object又在丟進,但這次的input是 wrapper, 是不是
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    for k, v in dataset[0].items():
        log('  {}: shape={}'.format(k, tuple(v.shape)))
        # 這裡的 dataset[0] 就對應到 class SRImplicitDownsampled 中的__getitem__() => return 一個dict
        #'inp': crop_lr, 'coord': hr_coord, 'cell': cell, 'gt': hr_rgb
        # 可以去看log
        """
        [DEBLUR]
            'sor_coord': sor_coord, #(2304 ,2)
            'crop_sor' : crop_sor,  #(3, 48, 48)
            'tar_rgb': tar_rgb,     #(2304 ,3)
            'deg_diff': deg_diff
        """

    # Dataloader 吃的是 class SRImplicitDownsampled object
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        shuffle=(tag == 'train'), num_workers= 16, pin_memory=True)
    return loader


def make_data_loaders():
    # 'train_dataset'與 'val_dataset' 的內容可以從 yaml檔去看 => args內有root_path (路徑), repeat, cache....
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])
        for _ in range(epoch_start - 1):
            lr_scheduler.step()
    else:
        model = models.make(config['model']).cuda()
        # 一般用法就是這樣，自行建立的 model class 都是繼承於 nn.Module class, 生出的object 會有一繼承來的 functions .parameters()
        # 再利用.parameters() 可以列舉出model中所有可以優化的參數 => 再丟進要創立optim用的func中
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler = None
        else:
            lr_scheduler = MultiStepLR(optimizer, **config['multi_step_lr'])

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, lr_scheduler


def train(train_loader, model, optimizer):
    model.train()
    loss_fn = nn.L1Loss()
    train_loss = utils.Averager()
    # train_loss = utils.Averager() => class 下的 object
    data_norm = config['data_norm']
    t = data_norm['inp'] # data_norm = {'inp': {'sub': [0.5], 'div': [0.5]}, 'gt': {'sub': [0.5], 'div': [0.5]}}
    sor_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda() # inp_sub = tensor([[[[0.5000]]]], device='cuda:0')
    sor_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda() # inp_div = tensor([[[[0.5000]]]], device='cuda:0')
    t = data_norm['gt']
    tar_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda() # gt_sub = tensor([[[[0.5000]]]], device='cuda:0')
    tar_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda() # gt_div = tensor([[[[0.5000]]]], device='cuda:0')

    for batch in tqdm(train_loader, leave=False, desc='train'):
        # batch 為從 class SRImplicitDownsampled return 出來的 dict, 透過dataloader去用idx取的
        """
        batch:
            'sor_coord': sor_coord, #(2304 ,2)
            'crop_sor' : crop_sor,  #(3, 48, 48)
            'tar_rgb': tar_rgb,     #(2304 ,3)
            'deg_diff': deg_diff
        """
        # 把batch 裡面那一包，都丟進GPU
        for k, v in batch.items():
            batch[k] = v.cuda()

        # 對inp做normalize
        crop_sor = (batch['crop_sor'] - sor_sub) / sor_div

        # 把normalized 的inp + coord + cell 丟進LIIF model
        # pred: (16, 2304, 3) => coord中2304個隨機座標所對應的預測value
        pred = model(crop_sor, batch['sor_coord'], batch['deg_diff'])

        # gt 其實也是影像 => 也去做normalize
        gt = (batch['tar_rgb'] - tar_sub) / tar_div
        loss = loss_fn(pred, gt)
        # 計算這一次的epoch中，每一次minibatch loss的平均
        train_loss.add(loss.item())

        # ~~~~wand~~~~
        wandb.log({"train_batch_loss": loss.item()})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = None; loss = None


    # 這一個epoch 的平均loss => 每一個minibatch的loss總和平均
    return train_loss.item()


def main(config_, save_path):
    global config, log, writer
    # config is global now
    config = config_
    log, writer = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)
    # 在save folder裡面把這次操作紀錄存下來

    train_loader, val_loader = make_data_loaders()
    if config.get('data_norm') is None:
        config['data_norm'] = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }

    model, optimizer, epoch_start, lr_scheduler = prepare_training()

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']        #1000
    epoch_val = config.get('epoch_val')    #1
    epoch_save = config.get('epoch_save')  #100
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        # 這應該是tensorboardX 的紀錄
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        train_loss = train(train_loader, model, optimizer)
        if lr_scheduler is not None:
            lr_scheduler.step()

        log_info.append('train: loss={:.4f}'.format(train_loss))
        writer.add_scalars('loss', {'train': train_loss}, epoch)
        # ~~~~wand~~~~
        wandb.log({"train_epoch_loss": train_loss})

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        sv_file = {
            'model': model_spec,
            'optimizer': optimizer_spec,
            'epoch': epoch
        }
        # save_path: ./save/_train_edsr-baseline-liif/
        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth')) #這是存最新的

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                os.path.join(save_path, 'epoch-{}.pth'.format(epoch))) #存每個epoch

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            if n_gpus > 1 and (config.get('eval_bsize') is not None):
                model_ = model.module
            else:
                model_ = model
            val_res = eval_psnr(val_loader, model_,
                data_norm=config['data_norm'],
                eval_type=config.get('eval_type'),  #None
                eval_bsize=config.get('eval_bsize')) #None

            log_info.append('val: psnr={:.4f}'.format(val_res))
            writer.add_scalars('psnr', {'val': val_res}, epoch)
            # ~~~~wand~~~~
            wandb.log({"val_psnr": val_res})

            if val_res > max_val_v:
                max_val_v = val_res
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth')) # val進步時存的

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))
        writer.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default= "configs/train-div2k/train_edsr-baseline-liif.yaml")
    parser.add_argument('--name', default= "ex1")
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    wandb.init(name=args.name, project="Reblur_experiment")

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        # train_edsr-baseline-liif.yaml
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    # save_path = ./save/_train_edsr-baseline-liif
    main(config, save_path)
    # config 裡面的東西都是以dict的形式儲存, 其中congif這個大dict裡面有9個小dict, 包括
    # train_dataset, val_dataset, datanorm, ... model, optimizer...


