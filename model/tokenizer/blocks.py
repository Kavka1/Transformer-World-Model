from typing import List, Dict
import torch
import torch.nn as nn


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class UpSample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv  =   with_conv
        if self.with_conv:
            self.conv   =   nn.Conv2d(
                in_channels     =   in_channels,
                out_channels    =   in_channels,
                kernel_size     =   3,
                stride          =   1,
                padding         =   1
            )
    
    def forward(self, x: torch.Tensor)  ->  torch.Tensor:
        x   =   nn.functional.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.with_conv:
            x   =   self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(
        self,
        in_channels:    int,
        with_conv:      bool,
    ) -> None:
        super().__init__()
        self.with_conv  =   with_conv
        if self.with_conv:
            self.conv   =   nn.Conv2d(
                in_channels     =   in_channels,
                out_channels    =   in_channels,
                kernel_size     =   3,
                stride          =   2,
                padding         =   0
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x   = nn.functional.pad(x, pad, mode='constant', value=0)
            x   = self.conv(x)
        else:
            x   = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels:    int,
        out_channels:   int     =   None,
        conv_shortcut:  bool    =   False,
        dropout:        float,
        temb_channels:  int     =   512
    ) -> None:
        super().__init__()

        self.in_channels        =   in_channels
        self.out_channels       =   in_channels if out_channels is None else out_channels
        self.use_conv_shortcut  =   conv_shortcut

        self.norm1              =   nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1              =   nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels > 0:
            self.temp_proj  =   torch.nn.Linear(temb_channels, out_channels)

        self.norm2              =   nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout            =   nn.Dropout(p=dropout)
        self.conv2              =   nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut  =   nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut   =   nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor)  -> torch.Tensor:
        h   =   x
        h   =   self.norm1(x)
        h   =   swish(h)
        h   =   self.conv1(h)

        if temb is not None:
            h   +=  self.temp_proj(swish(temb))[:, :, None, None]

        h   =   self.norm2(h)
        h   =   swish(h)
        h   =   self.dropout(h)
        h   =   self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x   =   self.conv_shortcut(x)
            else:
                x   =   self.nin_shortcut(x)
        
        return x + h


class AttnBlock(nn.Module):
    def __init__(
        self,
        in_channels: int
    ) -> None:
        super().__init__()

        self.norm   =   nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q      =   nn.Conv2d(
            in_channels     =    in_channels,
            out_channels    =    in_channels,
            kernel_size     =   1,
            stride          =   1,
            padding         =   0
        )
        self.k      =   nn.Conv2d(
            in_channels     =   in_channels,
            out_channels    =   in_channels,
            kernel_size     =   1,
            stride          =   1,
            padding         =   0
        )
        self.v      =   nn.Conv2d(
            in_channels     =   in_channels,
            out_channels    =   in_channels,
            kernel_size     =   1,
            stride          =   1,
            padding         =   0
        )
        self.proj_head  =   nn.Conv2d(
            in_channels     =   in_channels,
            out_channels    =   in_channels,
            kernel_size     =   1,
            stride          =   1,
            padding         =   0
        )

    def forward(self, x: torch.Tensor)  -> torch.Tensor:
        h_  =   x
        h_  =   self.norm(h_)

        q   =   self.q(h_)
        k   =   self.k(h_)
        v   =   self.v(h_)

        b, c, h, w  =   q.shape

        q       =   q.reshape(b, c, h * w)      #   [batch_size, channel, height * width]
        q       =   q.permute(0, 2, 1)          #   [batch_size, height * width, channel]
        k       =   k.reshape(b, c, h * w)      #   [batch_size, channel, height * width]
        w_      =   torch.bmm(q, k)             #   q^T * k [batch_size, channel, channel]
        w_      =   w_ * (int(c) ** (-0.5))     #   W / sqrt(channel)
        w_      =   torch.softmax(w_, dim=2)    

        v       =   v.reshape(b, c, h * w)
        w_      =   w_.permute(0, 2, 1)                  
        h_      =   torch.bmm(v, w_)            #   v^T * W
        h_      =   h_.reshape(b, c, h, w)

        h_      =   self.proj_head(h_)
        return x + h_