import argparse
import os
import math
from functools import partial

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils


def batched_predict(model, sor_img, sor_coord, deg_diff, bsize):
    with torch.no_grad():
        """
        [REBLUR]
            sor_img: (1, 3, 720 ,1280)
            sor_coord: (1, 921600, 2)
            deg_diff: 8 
        """



        deg_diff = torch.full((sor_coord.shape[1], 1), float(deg_diff), dtype=float).float()
        deg_diff = deg_diff.unsqueeze(0).cuda()

        #input 丟進去 model裡面給 encoder處理，此時model 這個object裡面的變數 self.feat 就有運算結果 (return 沒用到)
        model.gen_feat(sor_img)
        n = sor_coord.shape[1] #: 921600
        ql = 0
        preds = []
        while ql < n:
            # 1st round: qr = 30000, 每次都會在往後取30000個位置來運算, 有點像是指定要取哪一段進入model運算, 最後的一round, qr = n 剛好取到底
            qr = min(ql + bsize, n)
            # coord, cell 再丟進來前有先做unsqueeze 所以多了一維 => (1, 360000, 2)
            pred = model.query_rgb(sor_coord[:, ql: qr, :], deg_diff)
            preds.append(pred)
            ql = qr
        #沿著第一維度(30000) 做cat => pred變成 (1, 360000, 3)
        pred = torch.cat(preds, dim=1)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,
              verbose=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None: #這個
        metric_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        metric_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_res = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()

        crop_sor = (batch['crop_sor'] - inp_sub) / inp_div
        if eval_bsize is None: #這個
            with torch.no_grad():
                pred = model(crop_sor, batch['sor_coord'], batch['deg_diff'])

        else:
            pred = batched_predict(model, crop_sor,
                batch['sor_coord'], batch['deg_diff'], eval_bsize)
        pred = pred * gt_div + gt_sub
        pred.clamp_(0, 1)

        if eval_type is not None: # reshape for shaving-eval
            ih, iw = batch['inp'].shape[-2:]
            s = math.sqrt(batch['coord'].shape[1] / (ih * iw))
            shape = [batch['inp'].shape[0], round(ih * s), round(iw * s), 3]
            pred = pred.view(*shape) \
                .permute(0, 3, 1, 2).contiguous()
            batch['gt'] = batch['gt'].view(*shape) \
                .permute(0, 3, 1, 2).contiguous()

        res = metric_fn(pred, batch['tar_rgb'])
        val_res.add(res.item(), crop_sor.shape[0])

        if verbose:
            pbar.set_description('val {:.4f}'.format(val_res.item()))

        break

    return val_res.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        verbose=True)
    print('result: {:.4f}'.format(res))
