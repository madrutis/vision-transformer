import torchvision
import torch
from torch import nn
import torch.optim as optim
import einops
from math import sqrt, log

class InputEmbedding(nn.Module):
    def __init__(self, patch_size, n_channels, latent_size, dropout, img_height, img_width, device):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.input_size = patch_size * patch_size * n_channels
        self.dropout = nn.Dropout(dropout)
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)
        self.device = device
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

        # Random initialization of [CLS] token that is prepended to linear projection vector
        self.class_token = nn.Parameter(torch.randn(1, 1, self.latent_size))

        # Positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.latent_size)).to(self.device)
    
    def forward(self, input_data):
        input = input_data.to(self.device)
        # input_data (batch_size, num_channels, height, width) -> (batch_size, num_patches, input_size)
        patches = einops.rearrange(
            input, 
            'b c (h h1) (w w1) -> b (h w) (h1 w1 c)',
            h1=self.patch_size, w1=self.patch_size
        )
        # Linear projection
        projection = self.linearProjection(patches)
        # Prepend [CLS] token
        cls_token = einops.repeat(self.class_token, '1 1 d -> b 1 d', b=input.shape[0])
        # (bsz, num_patches + 1, latent_size)
        projection = torch.cat((cls_token, projection), dim=1)
        # Add positional embedding
        pos_embed = einops.repeat(self.pos_embedding, '1 m d -> b m d', b=input.shape[0])

        projection += pos_embed
        projection = self.dropout(projection)
        return projection



class ScaledDotProduct(nn.Module):
    def __init__(self, latent_size, dropout, mask):
        super(ScaledDotProduct, self).__init__()
        pass

    def forward(self, queries, keys, values):
        pass

    
class MultiheadAttention(nn.Module):
    def __init__(self, latent_size, num_heads, dropout, mask=None):
        super(MultiheadAttention, self).__init__()
        self.latent_size = latent_size
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # latent_size must be divisible by num_heads
        self.d_k = latent_size // num_heads
        self.w_q = nn.Linear(latent_size, latent_size, bias=False)
        self.w_k = nn.Linear(latent_size, latent_size, bias=False)
        self.w_v = nn.Linear(latent_size, latent_size, bias=False)
        self.w_o = nn.Linear(latent_size, latent_size, bias=False)

    @staticmethod
    def attention(self, query, key, value, dropout=None):
        # (batch, num_heads, num_patches, d_k) @ (b num_heads d_k num_patches) = (batch, num_heads, num_patches, num_patches)
        attention_scores = (query @ einops.rearrange(key, 'b num_heads num_p d_k -> b num_heads d_k num_p')) / sqrt(self.d_k)
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        
        return (attention_scores @ value), attention_scores
                    
    def forward(self, queries, keys, values):
        # input is (batch, num_patches, latent_dim)
        query = self.w_q(queries)
        key = self.w_k(keys)
        value = self.w_v(values)

        # (batch, num_patches, latent_dim) --> (batch, latent_dim, num_heads, d_k) --> (batch, num_heads, num_patches, d_k)
        query = einops.rearrange(query, 'b num_p (d_k num_heads) -> b num_heads num_p d_k', num_heads=self.num_heads)
        key = einops.rearrange(key, 'b num_p (d_k num_heads) -> b num_heads num_p d_k', num_heads=self.num_heads)
        value = einops.rearrange(value, 'b num_p (d_k num_heads) -> b num_heads num_p d_k', num_heads=self.num_heads)

        # (batch, num_heads, num_patches, d_k)
        out, attention_scores = self.attention(query, key, value, self.dropout)
        
        # combine all of the heads together
        out = einops.rearrange(out, 'batch num_heads num_patches d_k -> batch num_patches (num_heads d_k)')

        return self.w_o(out)
    
attention = MultiheadAttention(768, 4, .2)

test_vector = torch.randn(16, 768, 768)

print(attention(test_vector, test_vector, test_vector).shape)

class Encoder(nn.Module):
    def __init__(self, latent_size, num_heads, dropout, mask=None):
        super(Encoder, self).__init__()
        pass
                    
    def forward(self, queries, keys, values):
       pass



class ViT(nn.Module):
    def __init__(self, batch_size):
        # do stuff here
        self.batch_size = batch_size
    def forward(self, x):
        ## continue implementation
        return x


             


