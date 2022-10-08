import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from AFSD.common.config import config
from AFSD.common.layers import Unit1D, Unit3D
from AFSD.prop_pooling.boundary_pooling_op import BoundaryMaxPooling

from models.encoder import Encoder, EncoderLayer, ConvLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding
num_classes = config['dataset']['num_classes']
freeze_bn = config['model']['freeze_bn']
freeze_bn_affine = config['model']['freeze_bn_affine']

layer_num = 6
conv_channels = 512
feat_t = 4000 // 4
mask = np.array([-0.5000, -0.5000, -0.5000,  0.0000, -0.5000, -0.5000])
mask = torch.from_numpy(np.expand_dims(np.tile(mask, (72, 1)), 0)).cuda()

class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)


class ProposalBranch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.boundary_max_pooling = BoundaryMaxPooling()

        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature


class CoarsePyramid(nn.Module):
    def __init__(self, feat_channels, frame_num=4000):
        super(CoarsePyramid, self).__init__()
        out_channels = conv_channels
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = frame_num
        self.layer_num = layer_num
        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[0],
                output_channels=out_channels,
                kernel_shape=[1, 1, 1],
                padding='spatial_valid',
                use_batch_norm=False,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))

        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[1],
                output_channels=out_channels,
                kernel_shape=[1, 1, 1],
                use_batch_norm=False,
                padding='spatial_valid',
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))
        for i in range(2, layer_num):
            self.pyramids.append(nn.Sequential(
                Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))

        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.loc_proposal_branch = ProposalBranch(out_channels, 512)
        self.conf_proposal_branch = ProposalBranch(out_channels, 512)

        self.prop_loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None
        )
        self.prop_conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None
        )

        self.center_head = Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

        self.priors = []
        t = feat_t
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            if t % 2 != 0:
                t = t // 2 + 1
            else:
                t = t // 2

    def forward(self, feat_dict, ssl=False):
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        prop_locs = []
        prop_confs = []
        trip = []
        x2 = feat_dict['Mixed_5c']
        x1 = feat_dict['Mixed_4f']
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # print("x1", x1.shape)
        # print("x2", x2.shape)
        batch_num = x1.size(0)
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                x = conv(x1)
                x = x.squeeze(-1).squeeze(-1)
            elif i == 1:
                x = conv(x2)
                x = x.squeeze(-1).squeeze(-1)
                x0 = pyramid_feats[-1]
                y = F.interpolate(x, x0.size()[2:], mode='nearest')
                pyramid_feats[-1] = x0 + y
            else:
                x = conv(x)
            pyramid_feats.append(x)

        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        frame_level_feat = F.interpolate(frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        frame_level_feat = self.deconv(frame_level_feat)
        trip.append(frame_level_feat.clone())
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i, feat in enumerate(pyramid_feats):
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)

            locs.append(
                self.loc_heads[i](self.loc_head(loc_feat))
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )
            confs.append(
                self.conf_head(conf_feat).view(batch_num, num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )
            t = feat.size(2)
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                priors = self.priors[i].expand(batch_num, t, 1).to(feat.device)
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                segments = torch.cat([
                    torch.round(l_segment - out_plen),
                    torch.round(l_segment + in_plen),
                    torch.round(r_segment - in_plen),
                    torch.round(r_segment + out_plen)
                ], dim=-1)

                decoded_segments = torch.cat(
                    [priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                     priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:]],
                    dim=-1)
                plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)

            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(loc_feat, frame_level_feat,
                                                                     segments, frame_segments)
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(conf_feat, frame_level_feat,
                                                                        segments, frame_segments)
            if i == 0:
                trip.extend([loc_prop_feat_.clone(), conf_prop_feat_.clone()])
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = loc_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_loc_prop = loc_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                start_conf_prop = conf_prop_feat_[:, :ndim, ].permute(0, 2, 1).contiguous()
                end_conf_prop = conf_prop_feat_[:, ndim:, ].permute(0, 2, 1).contiguous()
                if ssl:
                    return trip
            prop_locs.append(self.prop_loc_head(loc_prop_feat).view(batch_num, 2, -1)
                             .permute(0, 2, 1).contiguous())
            prop_confs.append(self.prop_conf_head(conf_prop_feat).view(batch_num, num_classes, -1)
                              .permute(0, 2, 1).contiguous())
            centers.append(
                self.center_head(loc_prop_feat).view(batch_num, 1, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)
        conf = torch.cat([o.view(batch_num, -1, num_classes) for o in confs], 1)
        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, num_classes) for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, prop_loc, prop_conf, center, priors, start, end, \
               start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, attn_mask=attn_mask)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Informer(nn.Module):
    VALID_ENDPOINTS = ("enc_embedding","a", 'Mixed_4f',"b", 'Mixed_5c')
    def __init__(self, enc_in, c_out, out_len,
                 factor=10, d_model=512, n_heads=8, e_layers=1, d_layers=2, d_ff=512,
                 dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                 output_attention=False, distil=True, mix=True, device=None, final_endpoint='projection',
                 name='Informer'):
        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(Informer, self).__init__()
        self.end_points = {}
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention
        self._final_endpoint = final_endpoint
        Attn = ProbAttention if attn == 'prob' else FullAttention

        end_point = "enc_embedding"
        self.end_points[end_point] = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        if self._final_endpoint == end_point:
            return

        end_point = "a"
        self.end_points[end_point] = EncoderLayer(
            AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                           d_model, n_heads, mix=False),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_4f"
        self.end_points[end_point] = nn.Conv1d(in_channels=4000, out_channels=1000, kernel_size=1, bias=True)
        if self._final_endpoint == end_point:
            return

        end_point = "b"
        self.end_points[end_point] = EncoderLayer(
            AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention),
                           d_model, n_heads, mix=False),
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation
        )
        if self._final_endpoint == end_point:
            return

        end_point = "Mixed_5c"
        self.end_points[end_point] = nn.Conv1d(in_channels=4000, out_channels=500, kernel_size=1, bias=True)
        if self._final_endpoint == end_point:
            return

    def build(self):
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])


    def extract_features(self, x):
        output_dict = {}
        for end_point in self.VALID_ENDPOINTS:
            if end_point == "enc_embedding":
                x = self.end_points["enc_embedding"](x)
                output_dict[end_point] = x
            elif end_point == "a":
                x, attn = self.end_points["a"](x, attn_mask=None)
                output_dict[end_point] = x
            elif end_point == "b":
                x, attn = self.end_points["b"](x, attn_mask=None)
                output_dict[end_point] = x
            elif end_point == "Mixed_4f":
                x_ = self.end_points["Mixed_4f"](x)
                output_dict[end_point] = x_.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
            elif end_point == "Mixed_5c":
                x_ = self.end_points["Mixed_5c"](x)
                output_dict[end_point] = x_.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
        return output_dict


class Informer_Backbone(nn.Module):
    def __init__(self, final_endpoint='Mixed_5c', name='Informer_Backbone'):
        super(Informer_Backbone, self).__init__()
        self._model = Informer(final_endpoint=final_endpoint, name=name, enc_in=30, c_out=30, d_model=832, n_heads=8, e_layers=1, d_layers=1, d_ff=512, factor=5,
                                 distil=False, dropout=0.05, attn="full", mix=False, output_attention=False, device=None, out_len=1)
        self._model.build()

    def forward(self, x):
        return self._model.extract_features(x)


class BDNet(nn.Module):
    def __init__(self):
        super(BDNet, self).__init__()
        self.coarse_pyramid_detection = CoarsePyramid([832, 832])
        self.reset_params()

        self.backbone = Informer_Backbone()
        self.scales = [1, 4, 4]
        self.boundary_max_pooling = BoundaryMaxPooling()


    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, proposals=None, ssl=False):
        x = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)
        feat_dict = self.backbone(x)
        # print(feat_dict["encoder"])
        # print(feat_dict.keys()) mixed4f and mixed5c
        if ssl:
            top_feat = self.coarse_pyramid_detection(feat_dict, ssl)
            decoded_segments = proposals[0].unsqueeze(0)
            plen = decoded_segments[:, :, 1:] - decoded_segments[:, :, :1] + 1.0
            in_plen = torch.clamp(plen / 4.0, min=1.0)
            out_plen = torch.clamp(plen / 10.0, min=1.0)
            frame_segments = torch.cat([
                torch.round(decoded_segments[:, :, :1] - out_plen),
                torch.round(decoded_segments[:, :, :1] + in_plen),
                torch.round(decoded_segments[:, :, 1:] - in_plen),
                torch.round(decoded_segments[:, :, 1:] + out_plen)
            ], dim=-1)
            anchor, positive, negative = [], [], []
            for i in range(3):
                bound_feat = self.boundary_max_pooling(top_feat[i], frame_segments / self.scales[i])
                # for triplet loss
                ndim = bound_feat.size(1) // 2
                anchor.append(bound_feat[:, ndim:, 0])
                positive.append(bound_feat[:, :ndim, 1])
                negative.append(bound_feat[:, :ndim, 2])

            return anchor, positive, negative
        else:
            loc, conf, prop_loc, prop_conf, center, priors, start, end, \
            start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop = \
                self.coarse_pyramid_detection(feat_dict)
            return {
                'loc': loc,
                'conf': conf,
                'priors': priors,
                'prop_loc': prop_loc,
                'prop_conf': prop_conf,
                'center': center,
                'start': start,
                'end': end,
                'start_loc_prop': start_loc_prop,
                'end_loc_prop': end_loc_prop,
                'start_conf_prop': start_conf_prop,
                'end_conf_prop': end_conf_prop
            }

