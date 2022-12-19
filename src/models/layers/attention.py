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
"""Attention Cell"""
import mindspore.nn as nn
from mindspore import ops

from src.models.layers.drop_path import DropPath1D
from src.models.layers.identity import Identity


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer((dim,))
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath1D(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.k = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.v = nn.Dense(in_channels=dim, out_channels=dim, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_drop)
        self.proj = nn.Dense(in_channels=dim, out_channels=dim)
        self.proj_drop = nn.Dropout(keep_prob=1.0 - proj_drop)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        B, N, C = x.shape
        q = ops.Reshape()(self.q(x), (B, N, self.num_heads, C // self.num_heads)) * self.scale
        q = ops.Transpose()(q, (0, 2, 1, 3))
        k = ops.Reshape()(self.k(x), (B, N, self.num_heads, C // self.num_heads))
        k = ops.Transpose()(k, (0, 2, 3, 1))
        v = ops.Reshape()(self.v(x), (B, N, self.num_heads, C // self.num_heads))
        v = ops.Transpose()(v, (0, 2, 1, 3))

        attn = ops.BatchMatMul()(q, k)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = ops.Reshape()(ops.Transpose()(ops.BatchMatMul()(attn, v), (0, 2, 1, 3)), (B, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
