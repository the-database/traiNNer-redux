# docres_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from traiNNer.utils.registry import ARCH_REGISTRY, TESTARCH_REGISTRY
import numbers
from einops import rearrange

##########################################################################
## Helpers & Layers
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))
    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
    def forward(self, x):
        return self.body(x)

##########################################################################
## Main Architecture
@ARCH_REGISTRY.register()
class DocRes(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[2,3,3,4],
        num_refinement_blocks=4,
        heads=[1,2,4,8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type='WithBias',
        dual_pixel_task=True 
    ):
        super(DocRes, self).__init__()

        # Auto-detect logic
        self.use_auto_coords = False
        embed_channels = inp_channels
        
        if inp_channels == 3:
            print("DocRes: inp_channels is 3. Enabling Auto-Coordinate Injection (Internal input = 5).")
            self.use_auto_coords = True
            embed_channels = 5 # 3 RGB + 2 Coords

        self.patch_embed = OverlapPatchEmbed(embed_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])

        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim), kernel_size=1, bias=bias)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x, *args, **kwargs):
        # -----------------------------------------------------------
        # Handle Padding for odd-sized images (validation/inference)
        # -----------------------------------------------------------
        h, w = x.shape[2], x.shape[3]
        factor = 8 # The architecture has 3 downsample levels (2^3 = 8)
        H, W = ((h + factor - 1) // factor) * factor, ((w + factor - 1) // factor) * factor
        pad_h = H - h
        pad_w = W - w

        # Apply padding if necessary
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), 'reflect')

        inp_img = x
        
        # Inject Coordinates if inputs are just images (now using Padded size)
        if self.use_auto_coords:
            b, c, hh, ww = inp_img.shape
            y_coords = torch.linspace(-1, 1, hh, device=inp_img.device)
            x_coords = torch.linspace(-1, 1, ww, device=inp_img.device)
            mesh_y, mesh_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack((mesh_x, mesh_y), dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
            inp_img = torch.cat([inp_img, coords], dim=1)

        # -----------------------------------------------------------
        # Main Architecture Pass
        # -----------------------------------------------------------
        x = self.patch_embed(inp_img)
        encoder_l1 = self.encoder_level1(x)
        x = self.down1_2(encoder_l1)

        encoder_l2 = self.encoder_level2(x)
        x = self.down2_3(encoder_l2)

        encoder_l3 = self.encoder_level3(x)
        x = self.down3_4(encoder_l3)

        latent = self.latent(x)

        x = self.up4_3(latent)
        x = torch.cat([x, encoder_l3], 1)
        x = self.reduce_chan_level3(x)
        x = self.decoder_level3(x)

        x = self.up3_2(x)
        x = torch.cat([x, encoder_l2], 1)
        x = self.reduce_chan_level2(x)
        x = self.decoder_level2(x)

        x = self.up2_1(x)
        x = torch.cat([x, encoder_l1], 1)
        x = self.reduce_chan_level1(x)
        x = self.decoder_level1(x)

        x = self.refinement(x)

        out = self.output(x)
        
        # -----------------------------------------------------------
        # Remove Padding (Crop back to original)
        # -----------------------------------------------------------
        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]
            
        return out

# --------- Base Version ---------
@ARCH_REGISTRY.register()
def docres_base(
    inp_channels: int = 3, 
    out_channels: int = 3,
    dim: int = 48,
    num_blocks: list[int] = [2,3,3,4],
    num_refinement_blocks: int = 4,
    heads: list[int] = [1,2,4,8],
    ffn_expansion_factor: float = 2.66,
    bias: bool = False,
    LayerNorm_type: str = "WithBias",
    dual_pixel_task: bool = True,
    scale=None,
) -> DocRes:
    return DocRes(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=LayerNorm_type,
        dual_pixel_task=dual_pixel_task
    )

# --------- Large Version ---------
@ARCH_REGISTRY.register()
def docres_large(
    inp_channels: int = 3,
    out_channels: int = 3,
    dim: int = 64,
    num_blocks: list[int] = [3,4,4,6],
    num_refinement_blocks: int = 6,
    heads: list[int] = [2,4,8,16],
    ffn_expansion_factor: float = 2.66,
    bias: bool = False,
    LayerNorm_type: str = "WithBias",
    dual_pixel_task: bool = True,
    scale=None,
) -> DocRes:
    return DocRes(
        inp_channels=inp_channels,
        out_channels=out_channels,
        dim=dim,
        num_blocks=num_blocks,
        num_refinement_blocks=num_refinement_blocks,
        heads=heads,
        ffn_expansion_factor=ffn_expansion_factor,
        bias=bias,
        LayerNorm_type=LayerNorm_type,
        dual_pixel_task=dual_pixel_task
    )
    
##########################################################################
## Discriminator
@TESTARCH_REGISTRY.register()
def docres_discriminator(
    inp_channels: int = 3,
    out_channels: int = 1,
    dim: int = 64,
    num_blocks: list[int] = [1,2,2,3],
    heads: list[int] = [1,2,4,8],
    ffn_expansion_factor: float = 2.0,
    bias: bool = False,
    LayerNorm_type: str = "WithBias"
):
    class Discriminator(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            in_c = inp_channels
            for b in num_blocks:
                layers.append(nn.Conv2d(in_c, dim, 3, stride=1, padding=1, bias=bias))
                layers.append(nn.LeakyReLU(0.2, inplace=True))
                in_c = dim
            layers.append(nn.Conv2d(dim, out_channels, 3, stride=1, padding=1, bias=bias))
            self.model = nn.Sequential(*layers)

        def forward(self, x):
            return self.model(x)

    return Discriminator()