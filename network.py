import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import copy
import ptwt
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
import math
from einops import rearrange, repeat
from timm.models.layers import DropPath
from shuffle_attention_1d import ShuffleAttention


class ResBlock1D(nn.Module):
    def __init__(
            self,
            channels,
            kernel_size=3
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.res = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(channels),
            nn.PReLU(),
            nn.Conv1d(channels, channels, kernel_size, stride=1, padding=padding),
            nn.InstanceNorm1d(channels),
        )

    def forward(self, x):
        identity = x
        x = self.res(x)
        x = torch.add(x, identity)
        return x


class PFEN1D(nn.Module):
    def __init__(self):
        super(PFEN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 8, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(8, 16, 3, 1, 1),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 48, 3, 1, 1),
            nn.PReLU(),
        )

    def forward(self, input):
        x = input
        x = self.conv1(x)
        x_repeat = input.repeat(1, 8, 1)

        x = torch.cat((x, x_repeat), 1)

        x = self.conv2(x)
        x = torch.cat((x, x_repeat), 1)

        return x


class PFFN1D(nn.Module):
    def __init__(self, in_chans):
        super(PFFN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_chans, in_chans // 2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv1d(in_chans // 2, in_chans // 4, 3, 1, 1),
            nn.PReLU()
        )


        self.conv2 = nn.Sequential(
            nn.Conv1d(in_chans, in_chans // 4, 1, 1, 0),
            nn.PReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_chans // 4, in_chans // 8, 3, 1, 1),
            nn.PReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_chans // 8, 1, 3, 1, 1),
            nn.PReLU()
        )

    def forward(self, input):
        b, c, seq_len = input.shape
        x1 = self.conv1(input)
        x = self.conv(x1)

        return x


class SSM(nn.Module):
    def __init__(self, d_model,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 dt_rank="auto",
                 dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 bias=False,
                 use_fast_path=True,  # Fused kernel options
                 layer_idx=None,
                 device=None,
                 dtype=None, ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.activation = "silu"
        self.act = nn.SiLU()
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states):
        b, c, seqlen = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b d l -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        x, z = xz.chunk(2, dim=1)
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        assert self.activation in ["silu", "swish"]
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        out = rearrange(out, "b l d -> b d l")
        return out


class PatchEmbed1D(nn.Module):
    """ Sequence to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input sequence channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x)
        x = x.permute(0, 2, 1)
        if self.norm is not None:
            x = self.norm(x)
        x = x.permute(0, 2, 1)

        return x


class PatchExpand1D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, L, C = x.shape
        x = self.expand(x)
        x = rearrange(
            x,
            'b l (p c)-> b (l p) c',
            p=self.dim_scale,
            c=C // self.dim_scale
        )

        x = self.norm(x)

        return x


class Mamba(nn.Module):  # MSC_Mamba
    def __init__(self,
                 chans=32,
                 length=180,
                 expand=2,
                 kernelsize=3,
                 drop_rate=0.1):
        super().__init__()
        self.laynorm1 = nn.LayerNorm(normalized_shape=length)
        self.linear1 = nn.Linear(length, expand * length, bias=True)
        self.Dconv = nn.Sequential(
            nn.Conv1d(chans, 2 * chans, kernel_size=kernelsize, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.DWconv1d = nn.Conv1d(
            in_channels=chans,
            out_channels=chans,
            groups=chans,
            bias=True,
            kernel_size=kernelsize,
            padding=(kernelsize - 1) // 2,
        )
        self.act = nn.SiLU()
        self.SSM = SSM(chans)
        self.laynorm2 = nn.LayerNorm(expand * length)
        self.linear2 = nn.Linear(expand * length, length, bias=True)
        self.dorp = nn.Dropout(drop_rate)

    def forward(self, x):
        residual = x
        x = self.laynorm1(x)
        x = self.linear1(x)
        x = self.Dconv(x)
        x, z = x.chunk(2, dim=1)
        x = self.act(self.DWconv1d(x))  # [8,32,1440]
        x = self.SSM(x)
        x = self.laynorm2(x)
        x = x * self.act(z)
        x = self.linear2(x)
        output = self.dorp(x) + residual
        return output  # self.tail_DWconv1d(output)


class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=18):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv1d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ELU(inplace=True))
        for ep in range(num_of_layers - 2):
            layers.append(
                nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm1d(features))
            layers.append(nn.ELU(inplace=True))
        layers.append(nn.Conv1d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class Res_Mspa_Mamba(nn.Module):  # MSC_Mamba
    def __init__(self,
                 chans=32,
                 length=180,
                 expand=2,
                 drop_rate=0.1):
        super().__init__()
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.resblock = ResBlock1D(channels=chans)
        self.Dconv_2 = PatchEmbed1D(2, chans, 2 * chans)
        self.Dconv_4 = PatchEmbed1D(4, chans, 4 * chans)
        self.deconv_2 = PatchExpand1D(2 * chans, 2)
        self.deconv_4 = PatchExpand1D(4 * chans, 4)
        self.mamba = Mamba(chans, length)
        self.mamba_2 = Mamba(2 * chans, length // 2, expand)
        self.mamba_4 = Mamba(4 * chans, length // 4, expand)
        self.tail_mamba = Mamba(chans, length, expand)
        self.mambaout = nn.Sequential(
            nn.ReflectionPad1d(1),
            nn.Conv1d(chans * 3, chans, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm1d(chans),
            nn.PReLU(),
            nn.ReflectionPad1d(1),
            nn.Conv1d(chans, chans, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm1d(chans),
            nn.PReLU(),
        )

    def forward(self, input):
        x = self.resblock(input)
        x = self.mamba(x)

        x = self.resblock(x)

        x_2 = self.Dconv_2(x)
        x_2 = self.pos_drop(x_2)
        x_2 = self.mamba_2(x_2).permute(0, 2, 1)
        x_2 = self.deconv_2(x_2).permute(0, 2, 1)

        x_4 = self.Dconv_4(x)
        x_4 = self.pos_drop(x_4)
        x_4 = self.mamba_4(x_4).permute(0, 2, 1)
        x_4 = self.deconv_4(x_4).permute(0, 2, 1)

        y = torch.cat([x, x_2, x_4], dim=1)

        y = self.mambaout(y)
        y = self.tail_mamba(y)
        return y


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.gn = nn.GroupNorm(channels // 2, channels // 2)
        self.cweight = Parameter(torch.zeros(1, channels // 2, 1))
        self.cbias = Parameter(torch.ones(1, channels // 2, 1))
        self.sweight = Parameter(torch.zeros(1, channels // 2, 1))
        self.sbias = Parameter(torch.ones(1, channels // 2, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),l
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1
        x_channel = x_1 * self.sigmoid(x_channel)
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),l
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),l
        x_spatial = x_0 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),l
        out = torch.cat([x_spatial, x_channel], dim=1)  # bs*G,c//G,l
        return out


class AdvancedWaveletHFEN(nn.Module):


    def __init__(self, input_chans=64, num_filters=64, groups=8):
        super().__init__()
        self.input_chans = input_chans

        self.initial_fusion = nn.Sequential(
            nn.Conv1d(input_chans * 2, num_filters, 3, padding=1, groups=groups),
            nn.PReLU(),
            nn.Conv1d(num_filters, num_filters, 3, padding=1, groups=groups),
            nn.PReLU()
        )

        self.encoder1 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters, 3, padding=1, stride=2, groups=groups),
            nn.PReLU()
        )

        self.encoder2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters * 2, 3, padding=1, stride=2, groups=groups),
            nn.PReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv1d(num_filters * 2, num_filters * 4, 3, padding=1, groups=groups),
            nn.PReLU(),
            SpatialAttention(num_filters * 4),
            nn.Conv1d(num_filters * 4, num_filters * 2, 3, padding=1, groups=groups),
            nn.PReLU()
        )

        self.decoder1 = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 2, num_filters, 3, stride=2, padding=1, output_padding=1, groups=groups),
            nn.PReLU()
        )

        self.decoder2 = nn.Sequential(
            nn.ConvTranspose1d(2 * num_filters, num_filters, 3, stride=2, padding=1, output_padding=1, groups=groups),
            nn.PReLU()
        )

        self.output = nn.Sequential(
            nn.Conv1d(num_filters, num_filters // 2, 3, padding=1, groups=groups),
            nn.PReLU(),
            nn.Conv1d(num_filters // 2, input_chans * 2, 3, padding=1, groups=groups)
        )

    def forward(self, CA, CD):
        x = torch.cat([CA, CD], dim=1)
        b, c, seq_len = x.shape
        x = self.initial_fusion(x)

        x1 = self.encoder1(x)  # L/2
        x2 = self.encoder2(x1)  # L/4

        x = self.bottleneck(x2)

        x = self.decoder1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder2(x)

        x = self.output(x)
        x1, x2 = torch.split(x, c // 2, dim=1)

        CA_out = torch.cat([CA, x2], dim=1)
        CD_out = torch.cat([CD, x1], dim=1)

        return CA_out, CD_out


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.head = PFEN1D()
        self.dncnn = DnCNN()
        self.wavelet = 'haar'
        self.net = Res_Mspa_Mamba(chans=64, length=1440)
        self.Net_low = Res_Mspa_Mamba(chans=128, length=720)
        self.Net_high = Res_Mspa_Mamba(chans=128, length=720)
        self.jiaohu = AdvancedWaveletHFEN(input_chans=64)
        self.tail = PFFN1D(192)


    def forward(self, input):
        noise_level = self.dncnn(input)
        concat_x = torch.cat([input, noise_level], dim=1)
        x = self.head(concat_x)
        y = self.net(x)

        coeffs = ptwt.wavedec(x, self.wavelet, level=1)
        cA1, cD1 = coeffs
        cA1_input, cD1_input = self.jiaohu(cA1, cD1)

        cA1_output = self.Net_low(cA1_input)
        cD1_output = self.Net_high(cD1_input)
        reconstructed_coeffs = (cA1_output, cD1_output)
        reconstructed_x = ptwt.waverec(reconstructed_coeffs, self.wavelet)
        x = torch.cat([reconstructed_x, y], dim=1)
        out = self.tail(x) + input
        return out