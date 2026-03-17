# SPDX-FileCopyrightText: Copyright (c) 2024, Contributors
# SPDX-License-Identifier: Apache-2.0

"""U-Net Neural Operator (UNO) — Ashiqur Rahman et al., 2022."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from solaris.core.meta import ModelMetaData
from solaris.core.module import Module
from solaris.models.fno import FNOBlock2d


class UNO(Module):
    """U-Net Neural Operator.

    Encoder-decoder architecture where each scale uses FNO blocks for
    spectral feature extraction, skip connections bridge encoder and decoder,
    and strided Conv2d / bilinear upsampling handle scale + channel transitions.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    hidden_channels : int
        Channel width at the first encoder level.
    n_levels : int
        Number of encoder/decoder levels (excluding bottleneck).
    modes : int
        Fourier modes used in every FNOBlock2d.
    channel_multiplier : int
        Channel width multiplied by this factor at each encoder level.
    """

    _meta = ModelMetaData(
        name="UNO",
        nvp_tags=["pde", "operator-learning", "fourier", "u-net"],
        amp=True,
    )

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        n_levels: int = 3,
        modes: int = 12,
        channel_multiplier: int = 2,
    ) -> None:
        super().__init__(meta=self._meta)

        # Channel widths per encoder level
        widths: List[int] = [
            hidden_channels * (channel_multiplier ** i) for i in range(n_levels)
        ]

        self.lift = nn.Conv2d(in_channels, widths[0], 1)

        # Encoder: FNO block at each level + strided conv to downsample & expand channels
        self.encoder_blocks = nn.ModuleList([
            FNOBlock2d(widths[i], modes, modes) for i in range(n_levels)
        ])
        self.downsample = nn.ModuleList([
            # strided conv simultaneously halves spatial dims and expands channels
            nn.Conv2d(widths[i], widths[i + 1], kernel_size=2, stride=2)
            if i < n_levels - 1
            else nn.Identity()
            for i in range(n_levels)
        ])

        # Bottleneck
        bottleneck_w = widths[-1] * channel_multiplier
        self.bottleneck = nn.Sequential(
            nn.Conv2d(widths[-1], bottleneck_w, 1),
            nn.GELU(),
            FNOBlock2d(bottleneck_w, modes, modes),
            nn.Conv2d(bottleneck_w, widths[-1], 1),
        )

        # Decoder (processes levels in reverse: n_levels-1 → 0)
        # At each decoder step: upsample bottleneck/previous → cat with skip → skip_conv → FNO block
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        for i in reversed(range(n_levels)):
            # After upsampling we have widths[i] channels (from level below or bottleneck).
            # Skip from encoder also has widths[i] channels → cat = widths[i]*2.
            # First decoder step (from bottleneck): bottleneck already outputs widths[-1].
            self.skip_convs.append(nn.Conv2d(widths[i] * 2, widths[i], 1))
            self.decoder_blocks.append(FNOBlock2d(widths[i], modes, modes))
            if i > 0:
                # upsample + compress channels back to widths[i-1]
                self.upsample.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        nn.Conv2d(widths[i], widths[i - 1], 1),
                    )
                )
            else:
                self.upsample.append(nn.Identity())

        self.proj = nn.Sequential(
            nn.Conv2d(widths[0], widths[0] * 4, 1),
            nn.GELU(),
            nn.Conv2d(widths[0] * 4, out_channels, 1),
        )

        self._capture_init_args(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_levels=n_levels,
            modes=modes,
            channel_multiplier=channel_multiplier,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor  shape (B, C, H, W)

        Returns
        -------
        torch.Tensor  shape (B, out_channels, H, W)
        """
        x = self.lift(x)

        # Encoder — collect skip features before downsampling
        skips: List[torch.Tensor] = []
        for enc, down in zip(self.encoder_blocks, self.downsample):
            x = enc(x)
            skips.append(x)       # skip has widths[i] channels
            x = down(x)           # x now has widths[i+1] channels (or same at last level)

        x = self.bottleneck(x)    # x has widths[-1] channels

        # Decoder — iterate over levels from deepest to shallowest
        for skip_conv, dec, up, skip in zip(
            self.skip_convs,
            self.decoder_blocks,
            self.upsample,
            reversed(skips),
        ):
            # Align spatial dims if needed (odd input sizes)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = skip_conv(torch.cat([x, skip], dim=1))
            x = dec(x)
            x = up(x)

        return self.proj(x)
