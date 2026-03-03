import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        # time: tensor of shape (B,)
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()

        # Projection layer to match the time embedding to the channel dimension
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

    def forward(self, x, t_emb):
        x = self.act(self.bn1(self.conv1(x)))
        
        # Project time embedding and reshape to match image spatial dimensions
        time_emb = self.time_mlp(t_emb).unsqueeze(2).unsqueeze(3)
        
        # Add time context to the image features
        x = x + time_emb 
        
        x = self.act(self.bn2(self.conv2(x)))
        return x


def modulate(x, shift, scale):
    """Applies AdaLN modulation: x * (1 + scale) + shift"""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size)
        )
        
        # AdaLN parameters: predicts shift and scale for both norm layers from the time context
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # c is the time embedding
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(c).chunk(4, dim=1)
        
        # Self-Attention path
        x_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_modulated, x_modulated, x_modulated)
        x = x + attn_out
        
        # MLP path
        x_modulated = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + self.mlp(x_modulated)
        
        return x