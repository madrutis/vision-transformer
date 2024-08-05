import torchvision
import torch
from torch import nn
import torch.optim as optim
import einops
from math import sqrt, log

class Embedding(nn.Module):
    '''
    Constructs embedding for Vision Transformer
    Assumes photos are divisible by patch size

    '''
    def __init__(self, batch_size=32, patch_size=16, embed_dim=768, channels=3):
        super(self).__init__()
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.flat_patch_dim = patch_size * patch_size * channels

        # define projection
        self.projection = nn.Linear(self.flat_patch_dim, embed_dim)

    def forward(self, x):
        # TODO ADD CLASS TOKEN TO EMBEDDING

        # input is formatted in shape (batch_size, height, width, channels)
        # split image into patches and flatten (batch_size, (height / patch_size) * (width / patch_size), patch_size**2 * channels)
        patches = einops.rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        # include sqrt(embed_dim) as listed in the paper
        return self.projection(patches) * sqrt(self.embed_dim)

class PositionalEncoding(nn.Module):
    '''
    Constructs Positional Encoding for Vision Transformer
    Uses log in denominator of equation listed in paper (makes no difference to model training)
    '''
    def __init__(self, batch_size, patch_size, embed_dim, channels, width, height, dropout):
        super().__init__(self)
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (width / patch_size) * (height / patch_size)
        self.channels = channels
        self.dropout = nn.Dropout(dropout)

        # positional encodings need to be same shape as embedded input for addition 
        # Shape (b, h*w / p_s**2, embed_dim) = (b, num_patches, embed_dim)
        # log/divisor implementation taken from Umar Jamil on youtube
        self.positional_encoding = torch.zeros(self.num_patches, embed_dim)
        position = torch.arange(0, self.num_patches, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, embed_dim, 2).float() * -log(10000.0 / embed_dim))

        self.positional_encoding[:, 0::2] = torch.sin(position * divisor)
        self.positional_encoding[:, 1::2] = torch.cos(position * divisor)
        
        # repeat the positional encoding for every image/set of patches in the batch
        self.positional_encoding = einops.repeat(self.positional_encoding, 'num_patches embed_dim -> b num_patches embed_dim',b=batch_size)
        
        # save pe to the model parameters, even though it isn't learned
        self.register_buffer('positional_encoding', self.positional_encoding)

    
    def forward(self, x):
        # x is the batch of input embeddings in shape (b, num_patches, embed_dim)
        x = (x + self.positional_encoding).requires_grad_(False)
        return self.dropout(x)


class Attention(nn.Module):
    def __init__(self, batch_size):
        # do stuff here
        self.batch_size = batch_size
    def forward(self, x):
        ## continue implementation
        return x
    
class Multihead_Attention(nn.Module):
    def __init__(self, batch_size):
        # do stuff here
        self.batch_size = batch_size
    def forward(self, x):
        ## continue implementation
        return x

class ViT(nn.Module):
    def __init__(self, batch_size):
        # do stuff here
        self.batch_size = batch_size
    def forward(self, x):
        ## continue implementation
        return x


             


