import math
import random
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops.layers.torch import Rearrange

# 假设这些模块在你本地的其他文件里，需要保留导入
from subject_layers.Transformer_EncDec import Encoder, EncoderLayer
from subject_layers.SelfAttention_Family import FullAttention, AttentionLayer
from subject_layers.Embed import DataEmbedding
from debug_util import entropy  # 你代码中用到了 entropy 函数
from semantic_bridge_plugin import CrossModalBridgePlugin
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 核心配置与基础模块
# ==========================================

class Config:
    def __init__(self):
        self.task_name = 'classification'
        self.seq_len = 250
        self.pred_len = 250
        self.output_attention = False
        self.d_model = 250
        self.embed = 'timeF'
        self.freq = 'h'
        self.dropout = 0.25
        self.factor = 1
        self.n_heads = 4
        self.e_layers = 1
        self.d_ff = 256
        self.activation = 'gelu'
        self.enc_in = 63

class iTransformer(nn.Module):
    def __init__(self, configs, joint_train=False, num_subjects=10):
        super(iTransformer, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.enc_embedding = DataEmbedding(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, joint_train=False, num_subjects=num_subjects)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

    def forward(self, x_enc, x_mark_enc, subject_ids=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc, subject_ids)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        enc_out = enc_out[:, :63, :]
        return enc_out

class STAM(nn.Module):
    """
    终极优化版：移除会导致时域振铃的硬掩码频段注意力。
    专注于：频域特征驱动的通道注意力 + 时域卷积驱动的时间注意力。
    """
    def __init__(self, num_channels, seq_length, sampling_rate, bands=None, reduction=8):
        super().__init__()
        self.num_channels = num_channels
        self.seq_length = seq_length
        self.sampling_rate = sampling_rate
        
        # 1. 通道注意力 (Channel Attention)
        ch_hidden = max(1, num_channels // reduction)
        self.channel_fc1 = nn.Linear(num_channels, ch_hidden, bias=False)
        self.channel_fc2 = nn.Linear(ch_hidden, num_channels, bias=False)
        
        # 2. 时间注意力 (Temporal Attention)
        t_hidden = max(1, seq_length // reduction)
        self.temp_conv1 = nn.Conv1d(num_channels, num_channels, kernel_size=7, padding=3)
        self.temp_conv2 = nn.Conv1d(num_channels, num_channels, kernel_size=15, padding=7)
        self.temp_att_fc1 = nn.Linear(seq_length, t_hidden, bias=False)
        self.temp_att_fc2 = nn.Linear(t_hidden, seq_length, bias=False)

        # 3. 融合与归一化
        self.alpha = nn.Parameter(torch.zeros(2))
        self.norm  = nn.LayerNorm([num_channels, seq_length])

    def forward(self, x):
        # x: [B, C, T]
        B, C, T = x.shape
        
        # 分支 1: 基于频域感知的通道特征 (避免吉布斯振铃)
        Xf_abs = torch.fft.rfft(x, dim=-1).abs()  # [B, C, F]
        ch_desc = Xf_abs.mean(dim=-1)             # [B, C]
        
        ch_w = torch.sigmoid(self.channel_fc2(F.relu(self.channel_fc1(ch_desc))))  # [B, C]
        self.last_ch_w = ch_w.detach()
        ch_w_f = ch_w.unsqueeze(-1)  # [B, C, 1]
        
        x_spec = x * ch_w_f          # [B, C, T]

        # 分支 2: 时域分支
        t1 = F.gelu(self.temp_conv1(x))
        t2 = F.gelu(self.temp_conv2(x))
        x_temp = 0.5 * (t1 + t2)  # [B, C, T]
        
        t_desc = x_temp.mean(dim=1)  # [B, T]
        t_w = torch.sigmoid(self.temp_att_fc2(F.relu(self.temp_att_fc1(t_desc))))  # [B, T]
        
        x_temp = x_temp * t_w.unsqueeze(1) * ch_w_f  # [B, C, T]

        # 融合
        w = F.softmax(self.alpha, dim=0)  # [2]
        out = w[0] * x_spec + w[1] * x_temp
        out = self.norm(out)
        
        return out
        
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        super().__init__()
        self.tsconv = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), stride=(1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (63, 1), stride=(1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.tsconv(x)
        x = self.projection(x)
        return x

class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class SubjectLayers(nn.Module):
    """Per subject linear layer."""
    def __init__(self, in_channels: int, out_channels: int, n_subjects: int, init_id: bool = False):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_subjects, in_channels, out_channels))
        if init_id:
            assert in_channels == out_channels
            self.weights.data[:] = torch.eye(in_channels)[None]
        self.weights.data *= 1 / in_channels ** 0.5

    def forward(self, x, subjects):
        _, C, D = self.weights.shape
        weights = self.weights.gather(0, subjects.view(-1, 1, 1).expand(-1, C, D))
        return torch.einsum("bct,bcd->bdt", x, weights)

class FlattenHead(nn.Sequential):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class Enc_eeg(nn.Sequential):
    def __init__(self, emb_size=40, num_channels=63, seq_length=250, d_model=250, num_scales=5):
        super().__init__(
            PatchEmbedding(emb_size),
            FlattenHead()
        )

class Proj_eeg(nn.Sequential):
    def __init__(self, embedding_dim=1440, proj_dim=1024, drop_proj=0.3):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )

class STAMEncoder(nn.Module):
    def __init__(self, num_visual_classes=1654, num_channels=63, sequence_length=250, num_subjects=10, num_latents=1024):
        super(NeuralMCRL, self).__init__()
        default_config = Config()
        self.subject_layer = SubjectLayers(
            in_channels=num_channels,
            out_channels=num_channels,
            n_subjects=num_subjects,
            init_id=True
        )
        self.encoder = iTransformer(default_config)
        self.nsam = STAM(
            num_channels=num_channels,
            seq_length=sequence_length,
            sampling_rate=250.0
        )
        self.feature_norm = nn.LayerNorm([num_channels, sequence_length])
        self.enc_eeg = Enc_eeg()
        self.proj = nn.Linear(1440, 1024)
        self.proj_eeg = Proj_eeg()

    def forward(self, x, subject_ids, text_features=None, img_features=None, depth_features=None):
        x = self.subject_layer(x, subject_ids)
        x_trans = self.encoder(x, None, subject_ids)
        x_trans = self.nsam(x_trans)
        x_normalized = self.feature_norm(x_trans)
        enc_eeg = self.enc_eeg(x_normalized)
        z_eeg = self.proj(enc_eeg)
        eeg_features = self.proj_eeg(z_eeg)
        return z_eeg,eeg_features
