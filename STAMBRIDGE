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

class STNSAM_new(nn.Module):
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

# ==========================================
# 特征路由与注意力模块
# ==========================================

class MADRAttention(nn.Module):
    def __init__(self, dim, num_heads=16, modal_count=2, hidden=128, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim; self.H = num_heads; self.d = dim // num_heads
        self.modal_count = modal_count
        self.q_proj = nn.Linear(dim, dim)
        self.k_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(modal_count)])
        self.v_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(modal_count)])
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.route_mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.H * modal_count)
        )
        nn.init.zeros_(self.route_mlp[-1].bias)

    def forward(self, q, modal_kvs):
        B, M, D = q.shape
        Q = self.q_proj(q).view(B, M, self.H, self.d).transpose(1, 2)  # [B,H,M,d]

        if isinstance(modal_kvs, torch.Tensor):
            modal_list = [modal_kvs[:, i:i+1, :] for i in range(modal_kvs.shape[1])]
        else:
            modal_list = list(modal_kvs)

        actual_modal_count = len(modal_list)
        if actual_modal_count != self.modal_count:
            self.modal_count = actual_modal_count

        Ks, Vs = [], []
        for i, kv in enumerate(modal_list):
            Ni = kv.shape[1]
            K = self.k_projs[i](kv).view(B, Ni, self.H, self.d).transpose(1, 2)
            V = self.v_projs[i](kv).view(B, Ni, self.H, self.d).transpose(1, 2)
            Ks.append(K); Vs.append(V)
        
        pooled = q.mean(dim=1)  # [B,D]
        route_logits = self.route_mlp(pooled).view(B, self.H, self.modal_count)
        route_w = F.softmax(route_logits, dim=-1)  # [B,H,modal]
        self.last_route_w = route_w.detach()
        
        score_chunks = []
        for i in range(self.modal_count):
            K = Ks[i]  
            s = (Q @ K.transpose(-2, -1)) / (self.d ** 0.5)
            w = route_w[:, :, i].unsqueeze(-1).unsqueeze(-1)
            score_chunks.append(s * w)

        scores = torch.cat(score_chunks, dim=-1)  # [B,H,M,sumNi]
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)

        V_cat = torch.cat(Vs, dim=2)  # [B,H,sumNi,d]
        out = attn @ V_cat   # [B,H,M,d]
        out = out.transpose(1, 2).contiguous().view(B, M, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out, route_w, attn

class CrossModalBlock(nn.Module):
    def __init__(self, dim, n_heads=16, modal_count=2, dropout=0.1):
        super().__init__()
        self.attn = MADRAttention(dim, num_heads=n_heads, modal_count=modal_count)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim*4, dim),
            nn.Dropout(0.1),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, modal_kvs):
        q_norm = self.norm1(q)
        out, route_w, attn = self.attn(q_norm, modal_kvs)
        
        x = q + out
        x = self.norm2(x + self.dropout(self.ffn(x)))
        
        with torch.no_grad():
            attn_max = attn.max(dim=-1).values.mean()
            attn_ent = entropy(attn, dim=-1).mean()
            
        return x, route_w, attn_max.detach(), attn_ent.detach()

# ==========================================
# 顶层架构 (STAMBRIDGE)
# ==========================================

class NeuralMCRL(nn.Module):
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
        self.nsam = STNSAM_new(
            num_channels=num_channels,
            seq_length=sequence_length,
            sampling_rate=250.0
        )
        self.feature_norm = nn.LayerNorm([num_channels, sequence_length])
        self.enc_eeg = Enc_eeg()
        self.proj = nn.Linear(1440, 1024)

    def forward(self, x, subject_ids, text_features=None, img_features=None, depth_features=None):
        x = self.subject_layer(x, subject_ids)
        x_trans = self.encoder(x, None, subject_ids)
        x_trans = self.nsam(x_trans)
        x_normalized = self.feature_norm(x_trans)
        eeg_features = self.enc_eeg(x_normalized)
        eeg_features = self.proj(eeg_features)
        return eeg_features

class RouteModel(nn.Module):
    def __init__(self, sequence_length=250, num_subjects=10, embedding_dim=1024, proj_dim=1024):
        super(RouteModel, self).__init__()
        self.mcrl = NeuralMCRL(num_subjects=num_subjects, sequence_length=sequence_length)
        
        self.cross_layers_forward = nn.ModuleList([
             CrossModalBlock(dim=proj_dim, n_heads=8, modal_count=2) for _ in range(2)
        ])
        self.cross_layers_backward = nn.ModuleList([
             CrossModalBlock(dim=proj_dim, n_heads=16, modal_count=2) for _ in range(4)
        ])
        
        # Semantic mapping headers
        self.global_mlabel_train = nn.Sequential(
            nn.Linear(proj_dim, proj_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim//2, 1654)
        )
        self.global_mlabel_test = nn.Sequential(
            nn.Linear(proj_dim, proj_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(proj_dim//2, 200)
        )
        
        # Redundant projections
        self.proj_eeg1 = Proj_eeg()
        self.proj_eeg2 = Proj_eeg()
        self.proj_eeg3 = Proj_eeg()

    def forward_cross(self, q, kv):
        x = q
        rws = []; amax = []; aent = []
        for layer in self.cross_layers_forward:
            out = layer(x, kv)
            if isinstance(out, tuple):
                x_out, rw, amx, aen = out
                rws.append(rw.detach())
                amax.append(amx.detach())
                aent.append(aen.detach())
            else:
                x_out = out
            x = x_out
            
        dbg = {}
        if rws:
            R = torch.stack(rws, 0)  # [L,B,H,M]
            dbg["fw_route_w_mean"] = R.mean().item()
            dbg["fw_route_w_varH"] = R.var(dim=2).mean().item()
            dbg["fw_attn_max"]     = torch.stack(amax,0).mean().item()
            dbg["fw_attn_entropy"] = torch.stack(aent,0).mean().item()
        return x.squeeze(1), dbg
    
    def backward_cross(self, qs, kv):
        x = torch.cat(qs, dim=1)
        rws = []; amax = []; aent = []
        for layer in self.cross_layers_backward:
            out = layer(x, kv)
            if isinstance(out, tuple):
                x_out, rw, amx, aen = out
                rws.append(rw.detach())
                amax.append(amx.detach())
                aent.append(aen.detach())
            else:
                x_out = out
            x = x_out
            
        dbg = {}
        if rws:
            R = torch.stack(rws, 0)
            dbg["bw_route_w_mean"] = R.mean().item()
            dbg["bw_route_w_varH"] = R.var(dim=2).mean().item()
            dbg["bw_attn_max"]     = torch.stack(amax,0).mean().item()
            dbg["bw_attn_entropy"] = torch.stack(aent,0).mean().item()
        return x.mean(1), dbg

    def forward(self, eeg, subject_ids, img_features, text_features, use_gating=False):
        img_features = img_features.to(eeg.device)
        text_features = text_features.to(eeg.device)
    
        z_eeg = self.mcrl(eeg, subject_ids, None, None, None)  # [B, D]
        
        # Redundant projection heads
        z_eeg1 = self.proj_eeg1(z_eeg)
        z_eeg2 = self.proj_eeg2(z_eeg)
        z_eeg3 = self.proj_eeg3(z_eeg)
    
        eeg_img = z_eeg1
        eeg_text = z_eeg2
        eeg_dep = z_eeg3
    
        # Forward Routing (EEG query attending to Image & Text)
        kv = [img_features.unsqueeze(1), text_features.unsqueeze(1)]
        f1, dbg_fw = self.forward_cross((z_eeg.detach()).unsqueeze(1), kv)
        
        # Backward Routing (Image & Text queries attending to EEG)
        eeg_img_detach = z_eeg.detach()
        kv_eeg = [eeg_img_detach.unsqueeze(1), eeg_img_detach.unsqueeze(1)]
        f2, dbg_bw = self.backward_cross([img_features.unsqueeze(1), text_features.unsqueeze(1)], kv_eeg)
        
        # Dummy variables previously expected by your training loop return signature
        cycle_loss1 = cycle_loss2 = fuzzy_loss1 = fuzzy_loss2 = recon_eeg1 = 0
        dummy_gates = weight_reg = 0
        
        if self.training:
            semantic_logits = self.global_mlabel_train(f2)
        else:
            semantic_logits = self.global_mlabel_test(f2)
            
        return (z_eeg1, (eeg_img, eeg_text, eeg_dep), semantic_logits, 
                dummy_gates, weight_reg, cycle_loss1, cycle_loss2, 
                fuzzy_loss1, fuzzy_loss2, dbg_fw, dbg_bw, f1, f2, recon_eeg1)
