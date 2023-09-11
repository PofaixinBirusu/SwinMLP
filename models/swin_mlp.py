import torch
from torch import nn
from models.gmlp import gMLPBlock
from models.transformer import SelfAttention
from torch.utils import data
from torch.nn import functional as F
from models.pool import AttentivePool
from models.vit import ViT


class WMLP(nn.Module):
    def __init__(self, d_model, w_size):
        super(WMLP, self).__init__()
        seq_len = w_size ** 2
        self.w_size = w_size
        self.mlp = gMLPBlock(d_model, d_model*2, seq_len)

    def forward(self, x):
        # batch x c x h x w
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        w_size = self.w_size
        x = x.view(b, c, h, w // w_size, w_size).contiguous()
        x = x.permute([0, 1, 3, 2, 4]).contiguous()
        # b x c x h' x w' x w x w
        x = x.view(b, c, w // w_size, h // w_size, w_size, w_size).contiguous().permute([0, 1, 3, 2, 4, 5]).contiguous()

        x = x.permute([0, 2, 3, 4, 5, 1]).contiguous()
        x = x.view(b*(h//w_size)*(w//w_size), w_size*w_size, c).contiguous()

        x = self.mlp(x)
        x = x.view(b, h//w_size, w//w_size, w_size, w_size, c).contiguous()
        x = x.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, w//w_size, h, w_size]).contiguous()
        x = x.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, h, w).contiguous()

        return x


class SWMLP(nn.Module):
    def __init__(self, d_model, w_size):
        super(SWMLP, self).__init__()
        seq_len = w_size ** 2
        self.w_size = w_size
        self.mlp = gMLPBlock(d_model, d_model*2, seq_len)
        self.mlp_half = gMLPBlock(d_model, d_model*2, seq_len//2)
        self.mlp_quat = gMLPBlock(d_model, d_model*2, seq_len//4)

    def forward(self, x):
        # print(x.shape)
        # batch x c x h x w
        b, c, h, w = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        w_size = self.w_size

        # shift
        # b x c x (w // 2) x (w - w//2)
        top = x[:, :, 0:w_size//2, w_size//2:]
        # b x c x (h - w//2) x (w // 2)
        left = x[:, :, w_size//2:, 0:w_size//2]
        # b x c x w//2 x w//2
        left_top = x[:, :, :w_size//2, :w_size//2]
        x = torch.cat([x[:, :, w_size//2:, w_size//2:], top], dim=2)
        x = torch.cat([x, torch.cat([left, left_top], dim=2)], dim=3)

        # pachify
        x = x.view(b, c, h, w // w_size, w_size).contiguous()
        x = x.permute([0, 1, 3, 2, 4]).contiguous()
        # b x c x h' x w' x w x w
        x = x.view(b, c, w // w_size, h // w_size, w_size, w_size).contiguous().permute([0, 1, 3, 2, 4, 5]).contiguous()
        # b x h' x w' x w x w x c
        x = x.permute([0, 2, 3, 4, 5, 1]).contiguous()

        # 3 kinds of windows
        # first   b x (h'-1) x (w'-1) x w x w x c
        completed_windows = x[:, :h//w_size-1, :w//w_size-1, :, :, :]
        first_inp = completed_windows.contiguous().view(-1, w_size*w_size, c)
        # second  b x (h'-1) x (1) x w x w x c
        lr_windows = x[:, :h//w_size-1, w//w_size-1:, :, :, :]
        # second  b x (1) x (w'-1) x w x w x c
        ud_windows = x[:, h//w_size-1:, :w//w_size-1, :, :, :]
        # b x (h'-1) x (1) x w x w//2 x c    b x (h'-1) x (1) x w x w//2 x c
        l, r = lr_windows[:, :, :, :, :w_size//2, :], lr_windows[:, :, :, :, w_size//2:, :]
        # b x (1) x (w'-1) x w//2 x w x c    b x (1) x (w'-1) x w//2 x w x c
        u, d = ud_windows[:, :, :, :w_size//2, :, :], ud_windows[:, :, :, w_size//2:, :, :]
        second_inp = torch.cat([
            l.contiguous().view(-1, w_size*w_size//2, c),
            r.contiguous().view(-1, w_size*w_size//2, c),
            u.contiguous().view(-1, w_size*w_size//2, c),
            d.contiguous().view(-1, w_size*w_size//2, c)
        ], dim=0)
        # third  b x 1 x 1 x w x w x c
        quat_windows = x[:, h//w_size-1:, w//w_size-1:, :, :, :]
        third_inp = torch.cat([
            quat_windows[:, :, :, :w_size//2, :w_size//2, :].contiguous().view(-1, w_size*w_size//4, c),
            quat_windows[:, :, :, :w_size//2, w_size//2:, :].contiguous().view(-1, w_size*w_size//4, c),
            quat_windows[:, :, :, w_size//2:, :w_size//2, :].contiguous().view(-1, w_size*w_size//4, c),
            quat_windows[:, :, :, w_size//2:, w_size//2:, :].contiguous().view(-1, w_size*w_size//4, c)
        ], dim=0)

        # forward
        first_inp = self.mlp(first_inp)
        second_inp = self.mlp_half(second_inp)
        third_inp = self.mlp_quat(third_inp)

        # reset
        body = first_inp.view(b, h//w_size-1, w//w_size-1, w_size, w_size, c).contiguous()
        body = body.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, w//w_size-1, h-w_size, w_size]).contiguous()
        body = body.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, h-w_size, w-w_size).contiguous()

        l_num = b * (h // w_size - 1)
        l = second_inp[:l_num].view(b, h//w_size-1, 1, w_size, w_size//2, c).contiguous()
        # b c 1 h//w_size-1 w_size w_size//2
        l = l.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, 1, h-w_size, w_size//2]).contiguous()
        l = l.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, h-w_size, w_size//2).contiguous()

        r = second_inp[l_num:2*l_num].view(b, h//w_size-1, 1, w_size, w_size//2, c).contiguous()
        r = r.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, 1, h-w_size, w_size//2]).contiguous()
        r = r.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, h-w_size, w_size//2).contiguous()

        u = second_inp[2*l_num:3*l_num].view(b, 1, w//w_size-1, w_size//2, w_size, c).contiguous()
        # b c w//w_size-1 1 w_size//2 w_size
        u = u.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, w//w_size-1, w_size//2, w_size]).contiguous()
        u = u.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, w_size//2, w-w_size).contiguous()

        d = second_inp[3*l_num:].view(b, 1, w//w_size-1, w_size//2, w_size, c).contiguous()
        d = d.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, w//w_size-1, w_size//2, w_size]).contiguous()
        d = d.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, w_size//2, w-w_size).contiguous()

        quat_1 = third_inp[:b].view(b, 1, 1, w_size//2, w_size//2, c).contiguous()
        quat_1 = quat_1.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, 1, w_size//2, w_size//2]).contiguous()
        quat_1 = quat_1.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, w_size//2, w_size//2).contiguous()

        quat_2 = third_inp[b:2*b].view(b, 1, 1, w_size//2, w_size//2, c).contiguous()
        quat_2 = quat_2.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, 1, w_size//2, w_size//2]).contiguous()
        quat_2 = quat_2.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, w_size//2, w_size//2).contiguous()

        quat_3 = third_inp[2*b:3*b].view(b, 1, 1, w_size//2, w_size//2, c).contiguous()
        quat_3 = quat_3.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, 1, w_size//2, w_size//2]).contiguous()
        quat_3 = quat_3.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, w_size//2, w_size//2).contiguous()

        quat_4 = third_inp[3*b:].view(b, 1, 1, w_size//2, w_size//2, c).contiguous()
        quat_4 = quat_4.permute([0, 5, 2, 1, 3, 4]).contiguous().view([b, c, 1, w_size//2, w_size//2]).contiguous()
        quat_4 = quat_4.permute([0, 1, 3, 2, 4]).contiguous().view(b, c, w_size//2, w_size//2).contiguous()

        # merge
        x = torch.cat([torch.cat([body, l], dim=3), torch.cat([u, quat_1], dim=3)], dim=2)
        x = torch.cat([torch.cat([r, quat_2], dim=2), x], dim=3)
        x = torch.cat([torch.cat([quat_4, torch.cat([d, quat_3], dim=3)], dim=3), x], dim=2)

        return x


class SwinBlock(nn.Module):
    def __init__(self, d_model, w_size):
        super(SwinBlock, self).__init__()
        self.w_mlp = WMLP(d_model, w_size)
        self.sw_mlp = SWMLP(d_model, w_size)
        # self.sw_mlp = WMLP(d_model, w_size)

    def forward(self, x):
        # b x c x h x w
        x = self.w_mlp(x)
        x = self.sw_mlp(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class SwinMLP(nn.Module):
    def __init__(self,
                 in_chans=3,
                 dims=(128, 256, 512, 1024), depths=(3, 3, 27, 3),
                 w_sizes=(8, 8, 8, 10), img_token_dim=256):

        super(SwinMLP, self,).__init__()
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            # LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        # 对应stage2-stage4前的3个downsample
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2)
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks
        cur = 0
        # 构建每个stage中堆叠的block
        for i in range(4):
            stage = nn.Sequential(
                *[SwinBlock(dims[i], w_size=w_sizes[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.single_img_tokenizer = nn.Sequential(
            LayerNorm(dims[-1], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[-1], img_token_dim, kernel_size=1, stride=1)
        )
        # self.single_img_heatmap = nn.Sequential(
        #     LayerNorm(img_token_dim, eps=1e-6, data_format="channels_first"),
        #     nn.Conv2d(img_token_dim, img_token_dim, kernel_size=1, stride=1)
        # )
        #
        # self.softmax = nn.Softmax(dim=2)
        # self.pool = AttentivePool(dim=img_token_dim)

        # self.apply(self._init_weights)
        self.pool = ViT(image_size=384 // 32, patch_size=1, num_classes=img_token_dim, dim=img_token_dim,
                        channels=img_token_dim, heads=4, depth=2, mlp_dim=1024)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # b x c x h x w
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # b x c x h' x w'
        x = self.single_img_tokenizer(x)
        b, h_, w_ = x.shape[0], x.shape[2], x.shape[3]

        # # b c h' w'
        # w = self.single_img_heatmap(x)
        # # b h'w' c
        # w = w.view(b, x.shape[1], -1).permute([0, 2, 1])
        # w = w.mean(dim=2).view(b, 1, -1)
        # w = self.softmax(w)
        # w, x = self.pool(x.view(b, -1, h_*w_).permute([0, 2, 1]))
        # heatmaps = w.mean(dim=2).view(b, h_, w_)
        x, heatmaps = self.pool(x)
        # # # b x 1 x (h' x w')
        # # w = self.softmax(self.single_img_heatmap(x).view(b, 1, h_*w_))
        # # b x 1 x c
        # x = torch.matmul(w, x.view(b, -1, h_*w_).permute([0, 2, 1]))
        # x = x.squeeze(1)
        # # b x h' x w'
        # heatmaps = w.detach().squeeze(1).view(b, h_, w_)
        # b x c, b x h' x w'
        return x, heatmaps


class SwinMLP_ImgGroupDiagnosis(nn.Module):
    def __init__(self, n_class=1, in_chans=3,
                 dims=(128, 256, 512, 1024), depths=(3, 3, 27, 3),
                 w_sizes=(8, 8, 8, 10), img_token_dim=256,
                 batch_forward=False, forward_batch_size=32,
                 attn_layers=6):
        super(SwinMLP_ImgGroupDiagnosis, self).__init__()
        self.swin_mlp = SwinMLP(in_chans, dims, depths, w_sizes, img_token_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, img_token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.batch_forward = batch_forward
        self.forward_batch_size = forward_batch_size

        self.sa, self.n_sa_layer = nn.ModuleList(), attn_layers
        for _ in range(attn_layers):
            self.sa.append(SelfAttention(dim=img_token_dim, num_heads=8))
        self.final_norm = nn.LayerNorm(img_token_dim)

        self.cls_layer = nn.Linear(img_token_dim, n_class)

    def forward(self, x):
        # n x 3 x h x w
        feats, heatmaps = [], []

        if self.batch_forward:
            forward_num = x.shape[0] // self.forward_batch_size
            st, ed = 0, 0
            for i in range(forward_num):
                st, ed = i*self.forward_batch_size, (i+1)*self.forward_batch_size
                batch = x[st:ed, :, :, :]
                feat, heatmap = self.swin_mlp(batch)
                feats.append(feat)
                heatmaps.append(heatmap)
            if ed < x.shape[0]:
                batch = x[ed:, :, :, :]
                feat, heatmap = self.swin_mlp(batch)
                feats.append(feat)
                heatmaps.append(heatmap)
            feats, heatmaps = torch.cat(feats, dim=0), torch.cat(heatmaps, dim=0)
        else:
            # n x c, n x h' x w'
            feats, heatmaps = self.swin_mlp(x)
        # print(feats.shape)

        # (1 x n+1 x c)
        feats, attn_map = torch.cat([self.cls_token, feats], dim=0).unsqueeze(0), None
        for i in range(self.n_sa_layer):
            feats, attn_map = self.sa[i](feats)
        feats = self.final_norm(feats)
        # (1 x c), (n, )
        feats, imgs_weights = feats[0, 0:1], attn_map[0, 0, 1:].detach()

        # 1 x n_cls, (n, ), (n x h' x w')
        return self.cls_layer(feats), imgs_weights, heatmaps


if __name__ == '__main__':
    pass