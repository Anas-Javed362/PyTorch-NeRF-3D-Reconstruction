"""
Positional Encoding for NeRF.

Implements sinusoidal (Fourier feature) positional encoding that maps
low-dimensional inputs to higher-dimensional representations, allowing
the MLP to learn high-frequency functions of position and direction.

Reference:
    Mildenhall et al. "NeRF: Representing Scenes as Neural Radiance
    Fields for View Synthesis." ECCV 2020.
    https://arxiv.org/abs/2003.08934
"""

import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoder(nn.Module):
    """Sinusoidal positional encoder for NeRF inputs.

    Maps an input tensor of shape (..., D) to Fourier features of shape
    (..., out_dim) by applying sine and cosine at different frequencies.

    Given input dimension D and num_freqs L:
        - With include_input=True:  out_dim = D + 2 * L * D
        - With include_input=False: out_dim = 2 * L * D

    By default, frequencies are log-linearly spaced:
        freq_bands = [2^0, 2^1, ..., 2^(L-1)]

    Example:
        >>> encoder = PositionalEncoder(num_freqs=10, include_input=True)
        >>> x = torch.randn(1024, 3)
        >>> y = encoder(x)
        >>> y.shape
        torch.Size([1024, 63])   # 3 + 2*10*3 = 63

    Args:
        num_freqs: Number of frequency bands (L).
        include_input: If True, prepend the raw input x to the output.
        log_sampling: If True, use log-linear spacing 2^[0..L-1].
                      If False, use linear spacing [1..2^(L-1)].
    """

    def __init__(
        self,
        num_freqs: int,
        include_input: bool = True,
        log_sampling: bool = True,
    ) -> None:
        super().__init__()

        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling

        # Build frequency bands
        if log_sampling:
            # 2^0, 2^1, ..., 2^(L-1)
            freq_bands = 2.0 ** torch.arange(0, num_freqs, dtype=torch.float32)
        else:
            # Linearly spaced from 1 to 2^(L-1)
            freq_bands = torch.linspace(
                1.0, 2.0 ** (num_freqs - 1), num_freqs, dtype=torch.float32
            )

        # Register as a buffer so it moves with .to(device) and is saved in state_dict
        self.register_buffer("freq_bands", freq_bands)  # shape: (L,)

        # _input_dim is set lazily on the first forward pass; used to cache out_dim
        self._input_dim: Optional[int] = None
        self._out_dim: Optional[int] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def out_dim(self) -> Optional[int]:
        """Output dimension after encoding.

        Returns None until the encoder has seen its first input (because
        the input dimension D is not known at construction time).  After
        the first ``forward`` call, returns D + 2*L*D (or 2*L*D when
        include_input=False).
        """
        return self._out_dim

    def _compute_out_dim(self, input_dim: int) -> int:
        """Compute the output dimension for a given input dimension D."""
        enc_dim = 2 * self.num_freqs * input_dim
        if self.include_input:
            enc_dim += input_dim
        return enc_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor with sinusoidal features.

        Args:
            x: Input tensor of shape (..., D).

        Returns:
            Encoded tensor of shape (..., out_dim), where
            out_dim = D + 2*L*D (include_input=True) or 2*L*D.

        The encoding concatenates along the last dimension:
            [x,  sin(2^0 * x), cos(2^0 * x),
                 sin(2^1 * x), cos(2^1 * x),
                 ...
                 sin(2^(L-1) * x), cos(2^(L-1) * x)]
        """
        input_dim = x.shape[-1]

        # Cache out_dim on first call
        if self._input_dim is None:
            self._input_dim = input_dim
            self._out_dim = self._compute_out_dim(input_dim)
        elif self._input_dim != input_dim:
            raise ValueError(
                f"PositionalEncoder was first called with input_dim={self._input_dim}, "
                f"but now received input_dim={input_dim}."
            )

        parts = []

        if self.include_input:
            parts.append(x)  # (..., D)

        # freq_bands: (L,)  x: (..., D)
        # We want (..., D) * scalar -> (..., D) for each freq
        # Efficient: expand x to (..., 1, D) and freq_bands to (L, 1)
        # then broadcast to (..., L, D), flatten last two dims to (..., L*D)
        x_expanded = x.unsqueeze(-2)                        # (..., 1, D)
        freqs = self.freq_bands.view(*([1] * (x.dim() - 1)), -1, 1)  # (1,...,1, L, 1)
        scaled = freqs * x_expanded                          # (..., L, D)

        # Flatten the L and D dimensions
        *batch, L, D = scaled.shape
        scaled_flat = scaled.reshape(*batch, L * D)         # (..., L*D)

        parts.append(torch.sin(scaled_flat))                # (..., L*D)
        parts.append(torch.cos(scaled_flat))                # (..., L*D)

        return torch.cat(parts, dim=-1)                     # (..., out_dim)

    def extra_repr(self) -> str:
        return (
            f"num_freqs={self.num_freqs}, "
            f"include_input={self.include_input}, "
            f"log_sampling={self.log_sampling}"
        )


# ---------------------------------------------------------------------------
# Standalone helper
# ---------------------------------------------------------------------------

def encode(x: torch.Tensor, encoder: PositionalEncoder) -> torch.Tensor:
    """Apply a PositionalEncoder to an input tensor.

    Convenience wrapper so callers can write ``encode(x, encoder)``
    instead of ``encoder(x)``.

    Args:
        x: Input tensor of shape (..., D).
        encoder: A :class:`PositionalEncoder` instance.

    Returns:
        Encoded tensor of shape (..., out_dim).
    """
    return encoder(x)
