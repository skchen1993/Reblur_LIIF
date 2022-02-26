import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True):
        super().__init__()

        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode

        #生出前面的Feature extractor (Encoder) ex: EDSR
        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        #input 進 encoder 運算
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            # 原: feat: (1, 64, 256, 256)
            # Feature unfolding
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            # 後: feat: (1, 576, 256, 256)
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        # https://github.com/yinboc/liif/issues/34
        # rx, ry: the shortest distance between the center of each grid to its border
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) #feature (1, 576, 256, 256)
            # feat_coord 是由 input image size所建置的座標矩陣 (1, 2, 256, 256)

        # 不要忘記，coord_, coord 是由你目標大圖上的座標矩陣索取出來的部分座標值 (360000中的30000)
        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                 #for local ensemble, 每一個coord_中裝的都是欲查詢的座標，在第一round中，就先考慮 x-rx, y-ry, 等等在grid_sample()
                 #"nearest" model 下 就會找到離預查詢點左上方最近的實際存在於input image的座標

                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

                #   q_feat
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                # q_feature: (1, 30000, 576), rel_coord: (1, 30000, 2)
                # inp: (1, 30000, 578)
                if self.cell_decode:
                    # cell: (1, 30000, 2)
                    rel_cell = cell.clone()
                    # 將相對距離與image size脫鉤
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                #inp: (1, 30000, 580)
                bs, q = coord.shape[:2]
                # 這裡的 self.imnet 是 MLP, in_feature: 580
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                # pred: (1, 30000, 3) => RGB value
                preds.append(pred)
                # preds: list => size是4, 四個size為 (1,30000, 3)的tensor, 代表這 30000 個query 分別用offset + 看著的vector做的predict
                # offset的兩個維度剛好長寬，相乘得面積，再取絕對值
                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        # areas: list => size是4, 四個size為 (1,30000)的tensor, 代表這 30000 個query 與其左上右上左下右下目標監所圍成的面積
        # torch.stack(areas): 生出(4, 1, 30000)
        # tot_area: (1, 30000) => 每個QUERY看周圍鄰近四個pixel後的總面積
        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            #pred: (1, 30000, 3)
            #area: (1, 30000) => unsqueeze完 (1, 30000, 1) => broadcast
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
