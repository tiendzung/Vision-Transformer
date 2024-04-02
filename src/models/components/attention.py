import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, hidden_dim, numheads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, numheads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
    def forward(self, x):
        # x: [B, N, C]
        input_x = self.layer_norm_1(x)
        # input_x = [B, N, C]
        # query, key, value
        output_attn, attn_weight = self.attn(input_x, input_x, input_x)
        
        # output_attn = [B, N, C], attn_weight = [B, N, N]
        x = output_attn + x
        x = self.linear(self.layer_norm_2(x)) + x

        # x: [B, N, C]
        return x
