import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif')
class LIIF(nn.Module):
    # model = models[model_spec['name']](**model_args) 將dict當參數傳入，兩個東西就自動match前兩個input
    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True):
        super().__init__()

        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold


        #生出前面的Feature extractor (Encoder) ex: EDSR
        # 用 config['model'] 下的'args' 內的 'encoder_spec' 去建立 Encoder model
        self.encoder = models.make(encoder_spec)
        # 從config['model'] 帶進來for imnet的參數中只有out_dim = 3, 以及其他，沒有in_dim info, 所以要從encoder output出來的dim去做設定
        if imnet_spec is not None:
            #[REBLUR] imnet_in_dim = 64, 那乘13好像還好@@?
            imnet_in_dim = self.encoder.out_dim #here~
            if self.feat_unfold:
                imnet_in_dim *= 13 # [REBLUR] near_branch: feature unfolding + far_branch: look farther
            imnet_in_dim += 1 # [REBLUR] attach difference of deg

            # 用 config['model'] 下的'args' 內的 'imnet_spec' 去建立MLP model
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

    def gen_feat(self, inp):
        #input 進 encoder 運算
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, deg_diff):
        deg_diff_num = deg_diff[0,0,0].item()
        feat = self.feat
        #[REBLUR] feat: (2, 64, 48, 48)

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret


        if self.feat_unfold:
            # 原: feat: (2, c=64, 48, 48)
            # Feature unfolding
            feat_FU = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            #[REBLUR]feat_FU: (2, 576, 48, 48)

        feat_FU = feat_FU.view(feat_FU.shape[0], feat_FU.shape[1], -1).permute(0, 2, 1)
        # [REBLUR]feat_FU: (2, 2304, 9c=576)
        if self.local_ensemble:
            vx_lst = [-deg_diff_num, deg_diff_num]
            vy_lst = [-deg_diff_num, deg_diff_num]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0



        # field radius (global: [-1, 1])
        # https://github.com/yinboc/liif/issues/34
        # rx, ry: the shortest distance between the center of each grid to its border
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        """
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:]) #feature (1, 576, 256, 256)
            # feat_coord 是由 input image size所建置的座標矩陣 (1, 2, 256, 256)
        """



        far_branch = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                #for local ensemble, 每一個coord_中裝的都是欲查詢的座標，在第一round中，就先考慮 x-rx, y-ry, 等等在grid_sample()
                #"nearest" model 下 就會找到離預查詢點左上方最近的實際存在於input image的座標

                #[REBLUR], 分別往左上右上左下又下去看走deg_diff的位置去找nearest pixel
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6) #因此不會有超界的問題

                #[REBLUR] 利用coord_ (第一round) 去找出每個座標左下走deg_diff後, 離他最近的pixel vec
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)  #[REBLUR] q_feat : (2, 2304, c=64)
                far_branch.append(q_feat)

                """
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                """
                """
                # coord: HR 上的座標,  q_coord: 那些HR 上的座標所要看著的座標
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                # q_feature: (1, 30000, 576), rel_coord: (1, 30000, 2)
                # inp: (1, 30000, 578)
                """
                """
                if self.cell_decode:
                    # cell: (1, 30000, 2)
                    rel_cell = cell.clone()
                    # 將相對距離與image size脫鉤
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)
                """
        q_far = torch.cat((far_branch[0], far_branch[1], far_branch[2], far_branch[3]),2)
        # [REBLUR] q_far: (2, 2304, 4c=256)

        q_final = torch.cat((q_far, feat_FU, deg_diff), 2 )
        # [REBLUR] q_final: (2, 2304, 4c=256)
        bs, q = q_final.shape[:2]
        pred = self.imnet(q_final.view(bs * q, -1)).view(bs, q, -1)

        return pred

    def forward(self, crop_sor, sor_coord, deg_diff):
        self.gen_feat(crop_sor)
        return self.query_rgb(sor_coord, deg_diff)
