"""AsymmetricFFN — port of blocks.py:AsymmetricFFN, minus the mmcv registry.

Two details here are load-bearing and both are easy to get wrong by reading the code
the way it looks rather than the way it runs.

1. **The residual uses the PRE-NORMED input.** `identity = x` executes *after*
   `x = self.pre_norm(x)`, so the skip connection carries the normalised tensor, not the
   raw input. This is the same trap that cost real time in the Sparse4D port (see that
   project's bug log, finding #2).

2. **`identity_fc` is a Linear, not an Identity.** The constructor reads:

       for _ in range(num_fcs - 1):
           ...
           in_channels = feedforward_channels      # <- LOCAL rebind
       ...
       self.identity_fc = Identity() if in_channels == embed_dims else Linear(...)

   By the time that ternary runs, the local `in_channels` has been rebound to
   `feedforward_channels` (512), so it is compared against `embed_dims` (256) and the
   test fails — yielding `Linear(self.in_channels, embed_dims)` = `Linear(256, 256)`.
   With `num_fcs >= 2` (asserted upstream) this is *always* a Linear. It carries weights,
   so guessing `Identity` here silently breaks the checkpoint load.

Reproduced faithfully, quirk included — the checkpoint was trained with it.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AsymmetricFFN(nn.Module):
    def __init__(
        self,
        in_channels: int | None = None,
        pre_norm: bool = True,
        embed_dims: int = 256,
        feedforward_channels: int = 1024,
        num_fcs: int = 2,
        ffn_drop: float = 0.0,
        dropout_layer: dict | None = None,
        add_identity: bool = True,
    ) -> None:
        super().__init__()
        assert num_fcs >= 2, f"num_fcs should be no less than 2. got {num_fcs}."

        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.add_identity = add_identity

        local_in = embed_dims if in_channels is None else in_channels
        self.pre_norm = nn.LayerNorm(local_in) if pre_norm else None

        layers: list[nn.Module] = []
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(local_in, feedforward_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout(ffn_drop),
                )
            )
            local_in = feedforward_channels  # the rebind that decides identity_fc below
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        drop_prob = (dropout_layer or {}).get("drop_prob", 0.0)
        self.dropout_layer = nn.Dropout(drop_prob) if dropout_layer else nn.Identity()

        if add_identity:
            # See docstring note 2: `local_in` is feedforward_channels here, so this is
            # a Linear in every realistic configuration.
            self.identity_fc = (
                nn.Identity()
                if local_in == embed_dims
                else nn.Linear(
                    embed_dims if self.in_channels is None else self.in_channels,
                    embed_dims,
                )
            )

    def forward(self, x: torch.Tensor, identity: torch.Tensor | None = None) -> torch.Tensor:
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x  # NOTE: the PRE-NORMED x, not the original input
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
