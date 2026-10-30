"""TimeEmb + a frequency-referenced temporal correction branch.

Frequency path: copied from the supplied TimeEmb/models/TimeEmb.py.
Inverted attention and gate functions: adapted from FAiT (see README_FRCA.md).
This is a temporal-patch implementation, NOT a reproduction of FAiT's code.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class ReferenceAttention(nn.Module):
    """One attention layer; Q/K context and V content are explicit inputs."""

    def __init__(self, dim, heads, mode, beta_init, beta_fixed, dropout, inverted):
        super().__init__()
        if heads < 1 or dim < 1 or dim % heads:
            raise ValueError("frca_dim must be positive and divisible by frca_heads")
        if not 0 < beta_init < 1:
            raise ValueError("frca_beta_init must be strictly between 0 and 1")
        if beta_fixed != -1 and not 0 <= beta_fixed <= 1:
            raise ValueError("frca_beta_fixed must be -1 (learned), or in [0, 1]")
        self.heads, self.head_dim = heads, dim // heads
        self.mode, self.inverted = mode, inverted
        self.dropout = dropout
        self.qk_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        # Shared, bias-free projection keeps X/S subtraction in one coordinate system.
        self.v_proj = nn.Linear(dim, dim, bias=False)
        if mode == "complement" and beta_fixed == -1:
            self.beta_logits = nn.Parameter(torch.full((heads,), math.log(beta_init / (1 - beta_init))))
            self.register_buffer("fixed_beta", None)
        else:
            self.register_parameter("beta_logits", None)
            value = beta_fixed if mode == "complement" else 0.0
            self.register_buffer("fixed_beta", torch.full((heads,), value))
        self.gate_low = nn.Linear(dim, dim)
        self.gate_high = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.out_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * dim, dim), nn.Dropout(dropout))

    def beta(self):
        return self.beta_logits.sigmoid() if self.beta_logits is not None else self.fixed_beta

    def split_heads(self, z):
        # [B*C, patches, dim] -> [B*C, heads, patches, head_dim]
        return z.reshape(z.shape[0], z.shape[1], self.heads, self.head_dim).transpose(1, 2)

    def project_qkv(self, content_x, content_s, position):
        # Position is used for relations/gates, but NOT subtracted in V.
        if self.mode == "residual":
            content = content_x - content_s
        else:
            content = content_x
        context = content + position
        relation = self.qk_norm(context)
        q = self.split_heads(self.q_proj(relation))
        k = self.split_heads(self.k_proj(relation))
        v = self.split_heads(self.v_proj(content))
        if self.mode == "complement":
            v_reference = self.split_heads(self.v_proj(content_s))
            v = v - self.beta().view(1, self.heads, 1, 1) * v_reference
        return q, k, v, context

    def forward(self, content_x, content_s, position):
        q, k, v, context = self.project_qkv(content_x, content_s, position)
        # Non-causal attention is over OBSERVED historical patches only.
        low = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False)
        # Use the SAME attended result: low + high == v, including with dropout.
        high = v - low
        low = low.transpose(1, 2).reshape_as(content_x)
        high = high.transpose(1, 2).reshape_as(content_x)
        if self.inverted:
            g_low = self.gate_low(context).tanh()
            positive = F.softplus(self.gate_high(context))
            # Algebraically 2*p^2/(p^2+0.3678); this form avoids FP16 overflow.
            g_high = 2.0 * (positive / torch.hypot(positive, positive.new_tensor(math.sqrt(0.3678)))).square()
            z = g_low * low + g_high * high
        else:
            z = low
        z = self.out_dropout(self.out_proj(z))
        # No raw-X skip around the reference subtraction. FFN may learn a bias;
        # cancelling V does not imply the complete forecast is zero.
        return z + self.ff(self.ff_norm(z))


class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = configs.seq_len, configs.pred_len, configs.enc_in
        self.use_revin = bool(configs.use_revin)
        self.use_day_index, self.use_hour_index = bool(configs.use_day_index), bool(configs.use_hour_index)
        self.emb_len_hour, self.emb_len_day = configs.hour_length, configs.day_length
        if self.seq_len < 2 or self.pred_len < 1 or self.enc_in < 1:
            raise ValueError("seq_len >= 2, pred_len >= 1 and enc_in >= 1 are required")
        if self.emb_len_hour < 1 or self.emb_len_day < 1:
            raise ValueError("Calendar bank lengths must be positive")

        # Keep the original TimeEmb names/order to allow baseline weight loading.
        self.model = nn.Sequential(nn.Linear(self.seq_len, configs.d_model), nn.ReLU(),
                                   nn.Linear(configs.d_model, self.pred_len))
        self.emb_hour = nn.Parameter(torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1))
        self.emb_day = nn.Parameter(torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1))
        self.w = nn.Parameter(0.02 * torch.randn(1, self.seq_len))

        self.mode = getattr(configs, "frca_mode", "complement")
        if self.mode not in {"complement", "plain", "residual"}:
            raise ValueError("frca_mode must be complement, plain, or residual")
        self.reference = getattr(configs, "frca_reference", "timeemb")
        if self.reference not in {"timeemb", "moving_average"}:
            raise ValueError("Unknown frca_reference")
        self.detach_reference = bool(getattr(configs, "frca_detach_reference", 1))
        self.patch_len = getattr(configs, "frca_patch_len", 16)
        self.stride = getattr(configs, "frca_stride", 8)
        dim = getattr(configs, "frca_dim", 64)
        heads = getattr(configs, "frca_heads", 4)
        dropout = getattr(configs, "frca_dropout", 0.1)
        if not 1 <= self.stride <= self.patch_len:
            raise ValueError("Require 1 <= frca_stride <= frca_patch_len")
        if not 0 <= dropout < 1:
            raise ValueError("frca_dropout must be in [0, 1)")
        self.ma_kernel = getattr(configs, "frca_ma_kernel", 25)
        if self.ma_kernel < 1 or self.ma_kernel % 2 == 0:
            raise ValueError("frca_ma_kernel must be a positive odd number")
        self.num_patches = max(1, math.ceil((self.seq_len - self.patch_len) / self.stride) + 1)
        self.pad_right = (self.num_patches - 1) * self.stride + self.patch_len - self.seq_len
        self.patch_embed = nn.Linear(self.patch_len, dim, bias=False)
        self.position = nn.Parameter(torch.zeros(1, self.num_patches, dim))
        nn.init.normal_(self.position, std=0.02)
        # First version deliberately has ONE layer, so there are exactly heads betas.
        self.attention = ReferenceAttention(
            dim, heads, self.mode, getattr(configs, "frca_beta_init", 0.1),
            getattr(configs, "frca_beta_fixed", -1.0), dropout,
            bool(getattr(configs, "frca_inverted", 1)))
        self.time_head = nn.Linear(self.num_patches * dim, self.pred_len)
        self.eta = nn.Parameter(torch.full((1, 1, self.pred_len), getattr(configs, "frca_eta_init", 0.01)))

    def frequency_features(self, x, hour_index, day_index):
        """Original TimeEmb frequency computation. x and S: [B, C, L]."""
        spectrum = torch.fft.rfft(x, dim=2, norm="ortho")
        weight = torch.fft.rfft(self.w, dim=1, norm="ortho")
        real = spectrum.real
        if self.use_hour_index:
            if hour_index is None or hour_index.shape != (x.shape[0],):
                raise ValueError("hour_index must have shape [batch]")
            hour = self.emb_hour[hour_index.to(device=x.device, dtype=torch.long) % self.emb_len_hour]
            real = real - hour
        if self.use_day_index:
            if day_index is None or day_index.shape != (x.shape[0],) or (day_index < 0).any():
                raise ValueError("Day bank needs valid day indices; this dataset may return -1")
            day = self.emb_day[day_index.to(device=x.device, dtype=torch.long) % self.emb_len_day]
            real = real - day
        filtered = torch.complex(real, spectrum.imag) * weight
        real = filtered.real
        if self.use_day_index:
            real = real + day
        if self.use_hour_index:
            real = real + hour
        return torch.fft.irfft(torch.complex(real, filtered.imag), n=self.seq_len, dim=2, norm="ortho")

    def embed_history(self, x):
        # Replicate-pad only at the right edge, identically for X and S.
        if self.pad_right:
            x = F.pad(x, (0, self.pad_right), mode="replicate")
        patches = x.unfold(-1, self.patch_len, self.stride)
        return self.patch_embed(patches).reshape(-1, self.num_patches, self.position.shape[-1])

    def forward(self, x, hour_index=None, day_index=None, return_parts=False):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.seq_len, self.enc_in):
            raise ValueError("Expected x with shape [batch, seq_len, enc_in]")
        # FFT remains FP32 under mixed precision (arbitrary sequence lengths work).
        # Match original TimeEmb's unbiased variance, not an extra branch-wise RevIN.
        with torch.autocast(device_type=x.device.type, enabled=False):
            history = x.float()
            if self.use_revin:
                mean = history.mean(dim=1, keepdim=True)
                variance = history.var(dim=1, keepdim=True) + 1e-5
                history = (history - mean) / variance.sqrt()
            history = history.transpose(1, 2)
            s = self.frequency_features(history, hour_index, day_index)

        frequency_pred = self.model(s)
        reference = s
        if self.reference == "moving_average":
            padding = self.ma_kernel // 2
            reference = F.avg_pool1d(F.pad(history, (padding, padding), mode="replicate"), self.ma_kernel, stride=1)
        if self.detach_reference:
            reference = reference.detach()
        zx = self.embed_history(history)
        zs = self.embed_history(reference)
        time_features = self.attention(zx, zs, self.position)
        correction = self.time_head(time_features.flatten(1)).reshape(x.shape[0], self.enc_in, self.pred_len)
        output = (frequency_pred + self.eta * correction).transpose(1, 2)
        if self.use_revin:
            output = output * variance.sqrt() + mean
        if return_parts:
            # These diagnostics are in the COMMON NORMALIZED space, before denorm.
            return output, {"history": history, "frequency_features": s, "reference": reference, "frequency_pred": frequency_pred,
                            "correction": correction, "beta": self.attention.beta(), "eta": self.eta}
        return output

    def mechanism_state(self):
        return {"mode": self.mode, "reference": self.reference,
                "detach_reference": self.detach_reference, "heads": self.attention.heads,
                "layers": 1, "beta": self.attention.beta().detach().cpu().tolist(),
                "eta": self.eta.detach().cpu().flatten().tolist()}
