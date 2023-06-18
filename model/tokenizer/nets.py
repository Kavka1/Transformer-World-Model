"""
Quantization in VQGAN
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn

from Transformer_World_Model.model.tokenizer.blocks    import  ResnetBlock, AttnBlock, UpSample, DownSample, swish


@dataclass
class EncoderDecoderConfig:
    resolution:         int                 #   input resolutions
    in_channels:        int                 #   input channels
    z_channels:         int         
    ch:                 int                 #   channels
    ch_mult:            List[int]           #   multiplication of channels
    num_res_blocks:     int                 #   number of resnet block per layer
    attn_resolutions:   List[int]           #   resolutions requiring attention blocks
    out_ch:             int
    dropout:            int                 #   prob of dropout


class Encoder(nn.Module):
    def __init__(
        self,
        config: EncoderDecoderConfig
    ) -> None:
        super().__init__()

        self.config             =   config
        self.num_resolutions    =   len(config.ch_mult)
        temb_ch                 =   0       #   time step embedding channels

        # downsampling
        self.conv_in            =   nn.Conv2d(
            in_channels     =   config.in_channels,
            out_channels    =   config.ch,
            kernel_size     =   3,
            stride          =   1,
            padding         =   1
        )
        curr_res        =   config.resolution
        in_ch_mult      =   (1,) + tuple(config.ch_mult)
        self.down       =   nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block       =   nn.ModuleList()
            attn        =   nn.ModuleList()

            block_in    =   config.ch * in_ch_mult[i_level]
            block_out   =   config.ch * config.ch_mult[i_level]
            for i_block in range(self.config.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels     =   block_in,
                        out_channels    =   block_out,
                        temb_channels   =   temb_ch,
                        dropout         =   config.dropout
                    )
                )
                block_in    =   block_out
                if curr_res in config.attn_resolutions:
                    attn.append(
                        AttnBlock(
                            in_channels =   block_in
                        )
                    )

            down        =   nn.Module()
            down.block  =   block
            down.attn   =   attn
            if i_level != self.num_resolutions - 1:
                down.downsample     =   DownSample(block_in, with_conv=True)
                curr_res            =   curr_res // 2
            self.down.append(down)

        # middle
        self.mid            =   nn.Module()
        self.mid.block_1    =   ResnetBlock(
            in_channels     =   block_in,
            out_channels    =   block_in,
            temb_channels   =   temb_ch,
            dropout         =   config.dropout
        )
        self.mid.attn_1     =   AttnBlock(in_channels=block_in)
        self.mid.block_2    =   ResnetBlock(
            in_channels     =   block_in,
            out_channels    =   block_in,
            temb_channels   =   temb_ch,
            dropout         =   config.dropout
        )
        
        # end
        self.norm_out       =   nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out       =   nn.Conv2d(
            in_channels     =   block_in,
            out_channels    =   config.z_channels,
            kernel_size     =   3,
            stride          =   1,
            padding         =   1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temb    =   None

        # downsampling
        hs      =   [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h       =   self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h   =   self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h   =   hs[-1]
        h   =   self.mid.block_1(h, temb)
        h   =   self.mid.attn_1(h)
        h   =   self.mid.block_2(h, temb)

        # end
        h   =   self.norm_out(h)
        h   =   swish(h)
        h   =   self.conv_out(h)
        
        return h



class Decoder(nn.Module):
    def __init__(
        self,
        config: EncoderDecoderConfig
    ) -> None:
        super().__init__()

        self.config             =   config
        temb_ch                 =   0
        self.num_resolutions    =   len(config.ch_mult)

        # compute in_ch_multi, block_in and curr_res at lowest res
        in_ch_mult      =   (1,) + tuple(config.ch_mult)
        block_in        =   config.ch * config.ch_mult[self.num_resolutions - 1]
        curr_res        =   config.resolution // 2 ** (self.num_resolutions - 1)
        print(f"Tokenizer: shape of latent is {config.z_channels, curr_res, curr_res}")

        # z to block_in
        self.conv_in    =   nn.Conv2d(
            in_channels =   config.z_channels,
            out_channels=   block_in,
            kernel_size =   3,
            stride      =   1,
            padding     =   1
        )

        # middle
        self.mid        =   nn.Module()
        self.mid.block_1=   ResnetBlock(
            in_channels     =   block_in,
            out_channels    =   block_in,
            temb_channels   =   temb_ch,
            dropout         =   config.dropout
        )
        self.mid.attn_1 =   AttnBlock(
            in_channels     =   block_in,
        )
        self.mid.block_2=   ResnetBlock(
            in_channels     =   block_in,
            out_channels    =   block_in,
            temb_channels   =   temb_ch,
            dropout         =   config.dropout
        )

        # upsampling
        self.up     =   nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block   =   nn.ModuleList()
            attn    =   nn.ModuleList()
            block_out   =   config.ch * config.ch_mult[i_level]
            for i_block in range(config.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels     =   block_in,
                        out_channels    =   block_out,
                        temb_channels   =   temb_ch,
                        dropout         =   config.dropout
                    )
                )
                block_in    =   block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            
            up      =   nn.Module()
            up.block=   block
            up.attn =   attn
            if i_level != 0:
                up.upsample     =   UpSample(block_in, with_conv=True)
                curr_res        =   curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out   =   nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out   =   nn.Conv2d(in_channels=block_in, out_channels=config.out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb    =   None

        # z to block in
        h   =   self.conv_in(z)

        # middle
        h   =   self.mid.block_1(h, temb)
        h   =   self.mid.attn_1(h)
        h   =   self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.config.num_res_blocks + 1):
                h       =   self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h   =   self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h       =   self.up[i_level].upsample(h)
        
        # end
        h   =   self.norm_out(h)
        h   =   swish(h)
        h   =   self.conv_out(h)
        return h