# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import collections.abc
import math
from functools import partial
from itertools import repeat

import numpy as np
from mindspore import ParameterTuple
from mindspore import dtype as mstype
from mindspore import nn, Parameter, Tensor
from mindspore import ops as P
from mindspore.common import initializer as weight_init
from mindspore.common.initializer import Initializer as MeInitializer

from src.models.layers.attention import Block as OriginBlock
from src.models.layers.attention import Mlp
from src.models.layers.drop_path import DropPath1D
from src.models.layers.identity import Identity
from src.tools.var_init import KaimingUniform, _calculate_fan_in_and_fan_out


def assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


class GroupAttention(nn.Cell):
    """
    LSA: self attention within a group
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., ws=1):
        assert ws != 1
        super(GroupAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)
        self.batch_matmul = P.BatchMatMul(transpose_b=True)
        self.ws = ws

    def construct(self, x, H, W):
        B, N, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = P.Reshape()(x, (B, h_group, self.ws, w_group, self.ws, C)).transpose(0, 1, 3, 2, 4, 5)

        q = P.Reshape()(self.q(x), (B, total_groups, -1, self.num_heads, C // self.num_heads)).transpose(0, 1, 3, 2, 4)
        k = P.Reshape()(self.k(x), (B, total_groups, -1, self.num_heads, C // self.num_heads)).transpose(0, 1, 3, 2, 4)
        v = P.Reshape()(self.v(x), (B, total_groups, -1, self.num_heads, C // self.num_heads)).transpose(0, 1, 3, 2, 4)

        attn = self.batch_matmul(q, k) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = P.BatchMatMul()(attn, v).transpose(0, 1, 3, 2, 4).reshape(B, h_group, w_group, self.ws, self.ws, C)
        x = P.Reshape()(attn.transpose(0, 1, 3, 2, 4, 5), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention(nn.Cell):
    """
    GSA: using a  key to summarize the information for a group to be efficient.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=sr_ratio, stride=sr_ratio,
                                pad_mode='pad', has_bias=True)
            self.norm = nn.LayerNorm(normalized_shape=(dim,), epsilon=1e-05)

    def construct(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = P.Reshape()(P.Transpose()(x, (0, 2, 1,)), (B, C, H, W))
            x_ = self.sr(x_).reshape(B, C, -1).transpose(0, 2, 1)
            x_ = self.norm(x_)
            k = self.k(x_).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
            v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
        else:
            k = self.k(x).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)
            v = self.v(x).reshape(B, -1, self.num_heads, C // self.num_heads).transpose(0, 2, 1, 3)

        attn = P.BatchMatMul()(q, k.transpose(0, 1, 3, 2)) * self.scale
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = P.Reshape()(P.Transpose()(P.BatchMatMul()(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class SBlock(OriginBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super(SBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                                     drop_path, act_layer, norm_layer)

    def construct(self, x, H, W):
        return super(SBlock, self).construct(x, H, W)


class GroupBlock(OriginBlock):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, ws=1):
        super(GroupBlock, self).__init__(dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop,
                                         drop_path, act_layer, norm_layer)
        del self.attn
        if ws == 1:
            self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio)
        else:
            self.attn = GroupAttention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, ws)

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size,
                              pad_mode='pad', has_bias=True)
        self.norm = nn.LayerNorm(normalized_shape=(embed_dim,), epsilon=1e-05)

    def construct(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


# borrow from PVT https://github.com/whai362/PVT.git
class PyramidVisionTransformer(nn.Cell):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block, init_pos_cls_token=True):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embeds = nn.CellList()

        self.pos_drops = nn.CellList()
        self.blocks = nn.CellList()

        for i in range(len(depths)):
            if i == 0:
                self.patch_embeds.append(PatchEmbed(img_size, patch_size, in_chans, embed_dims[i]))
            else:
                self.patch_embeds.append(
                    PatchEmbed(img_size // patch_size // 2 ** (i - 1), 2, embed_dims[i - 1], embed_dims[i]))
            patch_num = self.patch_embeds[-1].num_patches + 1 if i == len(embed_dims) - 1 else self.patch_embeds[
                -1].num_patches

            self.pos_drops.append(nn.Dropout(keep_prob=1.0 - drop_rate))
        if init_pos_cls_token:
            pos_embeds = []
            for i in range(len(depths)):
                pos_embeds.append(Parameter(Tensor(np.zeros([1, patch_num, embed_dims[i]]), dtype=mstype.float32)))
            self.pos_embeds = ParameterTuple(pos_embeds)
            # init weights
            for pos_emb in self.pos_embeds:
                pos_emb.set_data(
                    weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02), pos_emb.shape, pos_emb.dtype))
            # cls_token
            self.cls_token = Parameter(Tensor(np.zeros([1, 1, embed_dims[-1]]), dtype=mstype.float32))

        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for k in range(len(depths)):
            _block = nn.CellList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
                sr_ratio=sr_ratios[k])
                for i in range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]

        self.norm = norm_layer((embed_dims[-1],))

        # classification head
        self.head = nn.Dense(in_channels=embed_dims[-1], out_channels=num_classes) if num_classes > 0 else Identity()

        self.init_weights()

    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(KaimingUniform(a=math.sqrt(5)),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Conv2d) and cell.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
                    if fan_in != 0:
                        bound = 1 / math.sqrt(fan_in)
                        cell.bias.set_data(weight_init.initializer(weight_init.Uniform(scale=bound),
                                                                   cell.bias.shape,
                                                                   cell.bias.dtype))

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            if i == len(self.depths) - 1:
                cls_tokens = self.cls_token.expand(B, -1, -1)
                x = P.Concat(1)((cls_tokens, x))
            x = x + self.pos_embeds[i]
            x = self.pos_drops[i](x)
            for blk in self.blocks[i]:
                x = blk(x, H, W)
            if i < len(self.depths) - 1:
                x = P.Transpose()(P.Reshape()(x, (B, H, W, -1,)), (0, 3, 1, 2,))

        x = self.norm(x)

        return x[:, 0]

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


# PEG  from https://arxiv.org/abs/2102.10882
class PosCNN(nn.Cell):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.SequentialCell([nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=s,
                                                 pad_mode='pad', padding=1, group=embed_dim, has_bias=True), ])
        self.s = s

    def construct(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = P.Reshape()(feat_token.transpose(0, 2, 1), (B, C, H, W,))
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(0, 2, 1)
        return x

    def no_weight_decay(self):
        return ['proj.%d.weight' % i for i in range(4)]


class RandomNormal(MeInitializer):
    def __init__(self, std=0.001):
        super(RandomNormal, self).__init__()
        self.std = std

    def _initialize(self, arr):
        std = self.std
        tmp = np.random.normal(0, std, arr.shape)
        assignment(arr, tmp)


class CPVTV2(PyramidVisionTransformer):
    """
    Use useful results from CPVT. PEG and GAP.
    Therefore, cls token is no longer required.
    PEG is used to encode the absolute position on the fly, which greatly affects the performance when input resolution
    changes during the training (such as segmentation, detection)
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], block_cls=Block):
        super(CPVTV2, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads, mlp_ratios,
                                     qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, depths,
                                     sr_ratios, block_cls, init_pos_cls_token=False)
        self.pos_block = nn.CellList([PosCNN(embed_dim, embed_dim) for embed_dim in embed_dims])
        self.init_weights()

    def init_weights(self):

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(RandomNormal(std=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def no_weight_decay(self):
        return set(['cls_token'] + ['pos_block.' + n for n, p in self.pos_block.named_parameters()])

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(len(self.depths)):
            x, (H, W) = self.patch_embeds[i](x)
            x = self.pos_drops[i](x)
            for j, blk in enumerate(self.blocks[i]):
                x = blk(x, H, W)
                if j == 0:
                    x = self.pos_block[i](x, H, W)  # PEG here
            if i < len(self.depths) - 1:
                x = x.reshape(B, H, W, -1).transpose(0, 3, 1, 2)

        x = self.norm(x)

        return P.ReduceMean()(x, 1)


class PCPVT(CPVTV2):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=SBlock):
        super(PCPVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                    mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                    norm_layer, depths, sr_ratios, block_cls)


class ALTGVT(PCPVT):
    """
    alias Twins-SVT
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256],
                 num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[4, 4, 4], sr_ratios=[4, 2, 1], block_cls=GroupBlock, wss=[7, 7, 7]):
        super(ALTGVT, self).__init__(img_size, patch_size, in_chans, num_classes, embed_dims, num_heads,
                                     mlp_ratios, qkv_bias, qk_scale, drop_rate, attn_drop_rate, drop_path_rate,
                                     norm_layer, depths, sr_ratios, block_cls)
        del self.blocks
        self.wss = wss
        # transformer encoder
        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.blocks = nn.CellList()
        for k in range(len(depths)):
            _block = nn.CellList([block_cls(
                dim=embed_dims[k], num_heads=num_heads[k], mlp_ratio=mlp_ratios[k], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i],
                norm_layer=norm_layer, sr_ratio=sr_ratios[k], ws=1 if i % 2 == 1 else wss[k]) for i in
                range(depths[k])])
            self.blocks.append(_block)
            cur += depths[k]
        self.init_weights()


def pcpvt_small_v0(**kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pcpvt_base_v0(**kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def pcpvt_large_v0(**kwargs):
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def alt_gvt_small(**kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[64, 128, 256, 512], num_heads=[2, 4, 8, 16], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 10, 4], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)
    return model


def alt_gvt_base(**kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[96, 192, 384, 768], num_heads=[3, 6, 12, 24], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    return model


def alt_gvt_large(**kwargs):
    model = ALTGVT(
        patch_size=4, embed_dims=[128, 256, 512, 1024], num_heads=[4, 8, 16, 32], mlp_ratios=[4, 4, 4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 18, 2], wss=[7, 7, 7, 7], sr_ratios=[8, 4, 2, 1],
        **kwargs)

    return model


if __name__ == "__main__":
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    data = Tensor(np.ones([2, 3, 224, 224]), dtype=mstype.float32)
    model = alt_gvt_base()
    param_nums = 0
    num = 0
    for name, param in model.parameters_and_names():
        print(name, param.shape)
        param_nums += np.prod(param.shape)
        num += 1
    print(model(data).shape)
    print(param_nums, num)
