import torch
import torch.nn as nn
from src.models.components.attention import TransformerBlock

# from attention import TransformerBlock

def image_to_patch(x, patch_size):
    # x: [B, C, H, W]
    B, C, H, W = x.shape

    # print(x.shape)
    # [B, C, H/P, P, W/P, P]
    x = x.reshape(
        B,
        C,
        torch.div(H, patch_size, rounding_mode='floor'),
        patch_size,
        torch.div(W, patch_size, rounding_mode='floor'),
        patch_size,
    )

    # print(x.shape)

    # [B, H/P, W/P, C, P, P]
    x = x.permute(0, 2, 4, 1, 3, 5)

    # print(x.shape)

    # [B, H * W / P^2, C, P, P]
    x  = x.flatten(1, 2)

    # print(x.shape)


    # [B, H * W / P^2, C * P^2] = [B, N, C * P^2]
    x = x.flatten(2, 4)

    # print(x.shape)


    return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        device = "cpu",
        dropout=0.0
    ):
        super().__init__()
        self.device = device
        self.patch_size = patch_size
        self.linear_projection = nn.Linear(num_channels * patch_size * patch_size, embed_dim)

        self.transformer = nn.Sequential(
            *(TransformerBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) 
              for _ in range(num_layers))
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

        self.cls_token = nn.Parameter(
            torch.rand(1, 1, embed_dim)
        )
        
        self.position_embeding = nn.Parameter(
            torch.rand(1, 1 + num_patches, embed_dim)
        )

        # inv_freq = 1.0 / (
        #     10000
        #     ** (torch.arange(0, embed_dim, 2, device = device).float() / embed_dim)
        # ) ## (embed_dim // 2,)

        # print(num_classes)
        
        # pos_enc_a = torch.sin(torch.stack([pos * inv_freq for pos in range(num_patches + 1)])) ## num_patches + 1, embed_dim // 2
        # pos_enc_b = torch.cos(torch.stack([pos * inv_freq for pos in range(num_patches + 1)])) ## num_patches + 1, embed_dim // 2

        # self.position_embeding = torch.cat([pos_enc_a, pos_enc_b], dim=-1).unsqueeze(0) ## 1, num_patches + 1, embed_dim

    def forward(self, x):
        # print(x.shape)
        x = image_to_patch(x, self.patch_size)
    
        B, N, D = x.shape
        # print(x.shape)
        x = self.linear_projection(x)
        
        cls_token = self.cls_token.repeat(B, 1, 1)

        x = torch.cat([cls_token, x], dim=1)
        x = x + self.position_embeding[:, : N + 1]

        x = self.dropout(x)
        x = x.transpose(0, 1) # N, B, D
        x = self.transformer(x) # N, B, D
        # x = x.transpose(0, 1)
        # print(x.shape)

        cls = x[0]
        output = self.mlp_head(cls)
        return output ## [B, num_classes]
    

if __name__ == "__main__":
    vit = VisionTransformer(
        embed_dim = 256,
        hidden_dim = 512,
        num_channels = 1,
        num_heads = 8,
        num_layers = 16,
        num_classes = 10,
        patch_size = 4,
        num_patches = 49,
        dropout = 0.2

    )
    x = torch.zeros(16, 1, 28, 28)
    print(vit(x).shape)

    # a = torch.arange(0, 3, 1).float()
    # # b = torch.Tensor([2, 4])
    # # print(torch.stack([a, b]))
    # print(torch.stack([i * a for i in range(5)]))