import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import ClipLoss
import random
from debug_util import StepGate,entropy,tstats,topk_neg_sim,hist_counts
# ==========================================
# 工具函数
# ==========================================
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x
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
        """
        q: [B, M, D]
        modal_kvs: either
            - a list of tensors, each [B, Ni, D], len == modal_count
            - OR a tensor [B, modal_count, D] (will be split into Ni=1 per modal)
        returns: out [B, M, D], route_w [B, H, modal_count]
        """
        B, M, D = q.shape
        Q = self.q_proj(q).view(B, M, self.H, self.d).transpose(1, 2)  # [B,H,M,d]

        # normalize input format
        if isinstance(modal_kvs, torch.Tensor):
            # [B, modal_count, D] -> list of [B,1,D]
            modal_list = [modal_kvs[:, i:i+1, :] for i in range(modal_kvs.shape[1])]
        else:
            modal_list = list(modal_kvs)

        # ensure modal_list length matches expected; if not, adapt modal_count locally
        actual_modal_count = len(modal_list)
        if actual_modal_count != self.modal_count:
            # adapt (route_mlp still sized for original modal_count; this is simpler
            # for now; in practice ensure modal_count match at construction)
            self.modal_count = actual_modal_count

        Ks, Vs = [], []
        for i, kv in enumerate(modal_list):
            # kv: [B, Ni, D]
            Ni = kv.shape[1]
            K = self.k_projs[i](kv)  # [B, Ni, D]
            V = self.v_projs[i](kv)
            K = K.view(B, Ni, self.H, self.d).transpose(1, 2)  # [B,H,Ni,d]
            V = V.view(B, Ni, self.H, self.d).transpose(1, 2)
            Ks.append(K); Vs.append(V)
        
        pooled = q.mean(dim=1)  # [B,D]
        route_logits = self.route_mlp(pooled).view(B, self.H, self.modal_count)

        route_w = F.softmax(route_logits ,dim=-1)  # [B,H,modal]
        self.last_route_w = route_w.detach() # 【新增这一行，保存模态路由权重】
        score_chunks = []
        for i in range(self.modal_count):
            K = Ks[i]  # [B,H,Ni,d]
            s = (Q @ K.transpose(-2, -1)) / (self.d ** 0.5)  # [B,H,M,Ni]
            w = route_w[:, :, i].unsqueeze(-1).unsqueeze(-1)  # [B,H,1,1]
            score_chunks.append(s * w)

        scores = torch.cat(score_chunks, dim=-1)  # [B,H,M,sumNi]
        attn = F.softmax(scores , dim=-1)
        attn = self.attn_drop(attn)

        V_cat = torch.cat([v for v in Vs], dim=2)  # [B,H,sumNi,d]
        out = attn @ V_cat   # [B,H,M,d]
        out = out.transpose(1, 2).contiguous().view(B, M, D)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out, route_w,attn
        
class CrossModalBlock(nn.Module):
    def __init__(self, dim, n_heads=16,modal_count=2,dropout=0.1):
        super().__init__()
        # multi‑head cross‑attention
        self.attn =  MADRAttention(dim, num_heads=n_heads)
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
        q = self.norm1(q)
        out, route_w,attn = self.attn(q, modal_kvs)  # out: [B,M,D], route_w [B,H,modal]
        if self.training and (random.random() < 0.01):
            with torch.no_grad():
                # route_w: [B,H,modal]
                rw_mean = route_w.mean(dim=0).cpu().numpy()  # mean per head across batch
                rw_var = route_w.var(dim=0).mean().item()
                # attn: [B,H,M,sumNi]
                attn_mean_per_head = attn.mean(dim=(0,2,3)).cpu().numpy()  # [H]
                attn_max_per_head = attn.max(dim=-1).values.mean(dim=(0,2)).cpu().numpy()  # [H]
                # res scale
                res_scale = float(self.res_scale.detach().cpu().item()) if hasattr(self,'res_scale') else None
        
                # print(f"[XBLOCK DBG] res_scale={res_scale} route_mean_heads={np.round(rw_mean.mean(axis=0),4) if rw_mean is not None else None}")
                # print(f"            attn_mean_per_head={np.round(attn_mean_per_head,4)}")
                # print(f"            attn_max_per_head={np.round(attn_max_per_head,4)}")
        x = q + out
        x = self.norm2(x + self.dropout(self.ffn(x)))
        with torch.no_grad():
        # attn: [B,H,M,sumNi]
            attn_max = attn.max(dim=-1).values.mean()        # 平均最大注意力
            attn_ent = entropy(attn, dim=-1).mean()          # 平均熵
            # print(f"[DEBUG] route_entropy: {-(route_w*route_w.log()).sum(-1).mean().item():.4f}, attn_entropy: {attn_ent:.4f}")
        return (x, route_w, attn_max.detach(), attn_ent.detach())   

class ConcatMLFFusion(nn.Module):
    def __init__(self, dim=1024, hidden=2048, out_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim*2),      # if concat f1,f2
            nn.Linear(dim*2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.LayerNorm(out_dim)
        )
    def forward(self, f1, f2):
        x = torch.cat([f1, f2], dim=-1)
        out = self.net(x)
        return F.normalize(out, dim=-1)

class StableMLPProj(nn.Module):
    """
    更稳健的 projection head：
    - 保留输入主成分 (residual) + 小 scale 校正，避免把信息完全重写为常向量
    - orthogonal init 减少早期奇异方向
    - 可选 LayerNorm（默认开启），在实验中可以把 norm=False 作为调试项
    """
    def __init__(self, dim, hidden_dim=None, use_norm=True, dropout=0.0, scale=0.08):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = max(256, dim // 2)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.use_norm = use_norm
        self.norm = nn.LayerNorm(dim) 
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # small residual scale: 保留输入主成分，默认 0.08（0.03-0.15 范围可调）
        self.scale = float(scale)

        # safer init
        nn.init.orthogonal_(self.fc1.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.orthogonal_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        out = norm( x + scale * (fc2(act(fc1(x)))) )
        这样能最大程度保留 x 的结构，仅学习小修正。
        """
        h = self.fc2(self.act(self.fc1(x)))
        out = x + self.scale * self.drop(h)
        out = self.norm(out)
        return out

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
class CrossModalBridgePlugin(nn.Module):
    """
    完全封装了原 RouteModel 中的双向交叉注意力和融合逻辑。
    可直接插入任何提取好特征的 Backbone 中使用。
    """
    def __init__(self, proj_dim=1024, forward_layers=2, backward_layers=4):
        super().__init__()
        self.proj_dim = proj_dim
        
        # 2. 后向特征交叉对齐层 (Img/Text 作为 Q，EEG 作为 K,V)
        self.cross_layers_backward = nn.ModuleList([
            CrossModalBlock(dim=proj_dim, n_heads=16, modal_count=2) for _ in range(backward_layers)
        ])
        # 3. 稳健的桥接投影与融合
        self.fmid2 = ConcatMLFFusion(dim=proj_dim, hidden=proj_dim * 2, out_dim=proj_dim)
        self.proj_eeg = Proj_eeg(embedding_dim=proj_dim, proj_dim=proj_dim)
        self.proj_f2 = Proj_eeg(embedding_dim=proj_dim, proj_dim=proj_dim)
        self.proj_f_mid_img2 = StableMLPProj(proj_dim, hidden_dim=proj_dim//4, use_norm=True, dropout=0.1, scale=0.08)
        self.loss_fn = ClipLoss()
        
    def backward_cross(self, qs, kv):
        x = torch.cat(qs, dim=1)
        
        for layer in self.cross_layers_backward:
            out = layer(x, kv)
            # 如果 layer 返回的是 tuple (x_out, rw, amx, aen)，只取第一个 x_out
            x = out[0] if isinstance(out, tuple) else out
            
        # 返回融合后的特征，空字典 {} 用于兼容外层的 f2, dbg_bw = ... 解包
        return x.mean(1)

    def forward(self, z_eeg, img_features, text_features, drop_rate, logit_scale_img):
        """
        输入: 均为对齐前或初步投影后的特征 [B, proj_dim]
        img_drop: 如果训练时外部做了 dropout 增强，可传入；否则默认使用 img_features
        返回: 
            f1 (前向对齐特征)
            f2 (后向对齐特征)
            f_mid_img2 (图像桥接监督特征)
            f_mid_eeg2 (脑电桥接监督特征)
            dbg_fw, dbg_bw (调试日志)
        """
        if logit_scale_img.dim() > 0:
            logit_scale_img = logit_scale_img.view(-1)[0]
        # 2. 后向特征交叉
        eeg_img_detach = z_eeg.detach()
        kv_eeg = [eeg_img_detach.unsqueeze(1), eeg_img_detach.unsqueeze(1)]
        f2 = self.backward_cross([img_features.unsqueeze(1), text_features.unsqueeze(1)], kv_eeg)
        
        # 3. 中继融合与双端映射
        f2_proj = F.normalize(self.proj_f2(f2), dim=-1)
        img_drop = F.dropout(img_features.detach(), p=drop_rate, training=True)
        f_mid_t2 = self.fmid2(img_drop, f2_proj)
        
        f_mid_img2 = F.normalize(self.proj_f_mid_img2(f_mid_t2), dim=-1)
        loss_img_bridge2 =  self.loss_fn(f_mid_img2, img_features.detach(), logit_scale_img)
        with torch.no_grad():
            f_mid_teacher_proj2 = F.normalize(self.proj_f_mid_img2(f_mid_t2))
        eeg_proj = self.proj_eeg(z_eeg)
        loss_student2 = self.loss_fn(eeg_proj, f_mid_teacher_proj2, logit_scale_img)
        
        return  f_mid_img2,loss_img_bridge2,loss_student2
