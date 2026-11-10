"""
Unit tests for positional encoding (src/encoding.py).
"""

import pytest
import torch
from src.encoding import PositionalEncoder, encode


def test_output_dim_with_input():
    """For D=3, L=10, include_input=True -> out_dim=63."""
    encoder = PositionalEncoder(num_freqs=10, include_input=True)
    x = torch.randn(16, 3)
    out = encoder(x)
    assert out.shape[-1] == 63
    assert encoder.out_dim == 63


def test_output_dim_without_input():
    """For D=3, L=10, include_input=False -> out_dim=60."""
    encoder = PositionalEncoder(num_freqs=10, include_input=False)
    x = torch.randn(16, 3)
    out = encoder(x)
    assert out.shape[-1] == 60
    assert encoder.out_dim == 60


def test_output_shape():
    """Verify output shape (B, out_dim) for batch input (B, D), including multi-batch dims."""
    encoder = PositionalEncoder(num_freqs=6, include_input=True)
    # D=3, L=6 -> 3 + 2 * 6 * 3 = 39
    B = 32
    x = torch.randn(B, 3)
    out = encoder(x)
    assert out.shape == (B, 39)

    # Multi-dimensional batch (e.g., [10, 20, 3])
    x_multi = torch.randn(10, 20, 3)
    out_multi = encoder(x_multi)
    assert out_multi.shape == (10, 20, 39)


def test_deterministic():
    """Same input produces identical output."""
    encoder = PositionalEncoder(num_freqs=8, include_input=True)
    x = torch.randn(25, 3)
    out1 = encoder(x)
    out2 = encoder(x)
    out3 = encode(x, encoder)
    assert torch.equal(out1, out2)
    assert torch.equal(out1, out3)


def test_frequency_range():
    """Verify that frequencies are powers of 2 for log_sampling=True."""
    L = 6
    encoder = PositionalEncoder(num_freqs=L, log_sampling=True)
    expected_freqs = torch.tensor([2.0 ** i for i in range(L)], dtype=torch.float32)
    assert torch.allclose(encoder.freq_bands, expected_freqs)


def test_gradient_flow():
    """Verify that gradients flow cleanly through the encoding."""
    encoder = PositionalEncoder(num_freqs=4, include_input=True)
    x = torch.randn(8, 3, requires_grad=True)
    out = encoder(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.all(x.grad == 0)


def test_different_input_dims():
    """Test positional encoding with D=1, D=2, and D=3."""
    # D=1, L=4, include_input=True -> 1 + 2 * 4 * 1 = 9
    enc1 = PositionalEncoder(num_freqs=4, include_input=True)
    x1 = torch.randn(10, 1)
    out1 = enc1(x1)
    assert out1.shape == (10, 9)
    assert enc1.out_dim == 9

    # D=2, L=5, include_input=False -> 2 * 5 * 2 = 20
    enc2 = PositionalEncoder(num_freqs=5, include_input=False)
    x2 = torch.randn(10, 2)
    out2 = enc2(x2)
    assert out2.shape == (10, 20)
    assert enc2.out_dim == 20

    # D=3, L=10, include_input=True -> 3 + 2 * 10 * 3 = 63
    enc3 = PositionalEncoder(num_freqs=10, include_input=True)
    x3 = torch.randn(10, 3)
    out3 = enc3(x3)
    assert out3.shape == (10, 63)
    assert enc3.out_dim == 63
