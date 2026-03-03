# Local folder imports
from .layers import UNetBlock, SinusoidalTimeEmbedding


class FlowMatchingUNet(nn.Module):
    def __init__(self, in_channels=3, base_dim=64):
        super().__init__()
        time_dim = base_dim * 4
        
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_dim),
            nn.Linear(base_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Initial Convolution
        self.init_conv = nn.Conv2d(in_channels, base_dim, 3, padding=1)

        # Downsampling (Encoder)
        self.down1 = UNetBlock(base_dim, base_dim * 2, time_dim)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = UNetBlock(base_dim * 2, base_dim * 4, time_dim)
        self.pool2 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = UNetBlock(base_dim * 4, base_dim * 4, time_dim)

        # Upsampling (Decoder)
        self.up1 = nn.ConvTranspose2d(base_dim * 4, base_dim * 2, 2, stride=2)
        self.up_block1 = UNetBlock(base_dim * 4, base_dim * 2, time_dim) # *4 because of skip connection

        self.up2 = nn.ConvTranspose2d(base_dim * 2, base_dim, 2, stride=2)
        self.up_block2 = UNetBlock(base_dim * 2, base_dim, time_dim) # *2 because of skip connection

        # Final Output (Predicts the velocity, so output channels match input channels)
        self.final_conv = nn.Conv2d(base_dim, in_channels, 1)

    def forward(self, x, t):
        # 1. Embed the time
        t_emb = self.time_embed(t)

        # 2. Initial processing
        x0 = self.init_conv(x)

        # 3. Encoder path (save outputs for skip connections)
        d1 = self.down1(x0, t_emb)
        d1_pooled = self.pool1(d1)
        
        d2 = self.down2(d1_pooled, t_emb)
        d2_pooled = self.pool2(d2)

        # 4. Bottleneck
        b = self.bottleneck(d2_pooled, t_emb)

        # 5. Decoder path (with skip connections)
        u1 = self.up1(b)
        u1 = torch.cat([u1, d2], dim=1) # The skip connection!
        u1 = self.up_block1(u1, t_emb)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d1], dim=1) # The skip connection!
        u2 = self.up_block2(u2, t_emb)

        # 6. Predict the velocity field
        velocity = self.final_conv(u2)
        return velocity

class FlowMatchingDiT(nn.Module):
    def __init__(self, in_channels=3, input_size=640, patch_size=16, hidden_size=256, depth=6, num_heads=8):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.patch_size = patch_size
        
        # Calculate sequence length based on image size and patch size
        self.grid_size = input_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # 1. Patchification (Conv2d with stride = patch_size)
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # 2. Time Embedding (Using the same logic as the U-Net)
        self.t_embedder = nn.Sequential(
            nn.Linear(1, hidden_size), # Simple linear projection for time
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 3. Learnable Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])
        
        # 5. Final Layer to predict the velocity patch
        self.final_layer = nn.Sequential(
            nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6),
            nn.Linear(hidden_size, patch_size * patch_size * in_channels)
        )

    def forward(self, x, t):
        """
        x: (B, C, H, W) intermediate noisy image
        t: (B,) time step
        """
        B, C, H, W = x.shape
        
        # Ensure input dimensions match our grid expectations
        assert H == self.input_size and W == self.input_size, f"Input size must be {self.input_size}"

        # 1. Patchify the image and flatten: (B, hidden_size, grid, grid) -> (B, num_patches, hidden_size)
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # 2. Embed the time step: (B,) -> (B, hidden_size)
        t = t.unsqueeze(1) # Make it (B, 1)
        c = self.t_embedder(t)
        
        # 3. Pass through DiT blocks
        for block in self.blocks:
            x = block(x, c)
            
        # 4. Final projection to velocity patches: (B, num_patches, patch_size^2 * C)
        x = self.final_layer(x)
        
        # 5. Un-patchify back to image dimensions: (B, C, H, W)
        velocity = self.unpatchify(x, B, C, H, W)
        
        return velocity

    def unpatchify(self, x, B, C, H, W):
        """Reshapes the sequence of patches back into a 2D image."""
        p = self.patch_size
        h = w = self.grid_size
        
        # Reshape to (B, grid_h, grid_w, patch_size, patch_size, C)
        x = x.reshape(shape=(B, h, w, p, p, C))
        
        # Permute to (B, C, grid_h, patch_size, grid_w, patch_size)
        x = torch.einsum('nhwpqc->nchpwq', x)
        
        # Flatten the grid and patches into final image dimensions
        velocity = x.reshape(shape=(B, C, H, W))
        return velocity