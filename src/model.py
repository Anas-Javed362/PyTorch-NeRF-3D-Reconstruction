"""
NeRF Neural Network Model.

Implements the MLP architecture from:
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    Mildenhall et al., ECCV 2020.  https://arxiv.org/abs/2003.08934

The network takes sinusoidally-encoded 3D positions (and optionally encoded
view directions) and outputs RGB colour + volume density (sigma).

Typical usage
-------------
>>> from src.encoding import PositionalEncoder
>>> from src.model import NeRF, create_nerf, count_parameters
>>>
>>> config = {
...     "pos_enc_freqs": 10,
...     "dir_enc_freqs": 4,
...     "num_layers": 8,
...     "hidden_dim": 256,
...     "skip_connect": [4],
...     "use_viewdirs": True,
... }
>>> pos_encoder, dir_encoder, model = create_nerf(config)
>>> # Encode inputs
>>> pts = torch.randn(1024, 3)
>>> dirs = torch.randn(1024, 3)
>>> x_enc = pos_encoder(pts)
>>> d_enc = dir_encoder(dirs)
>>> out = model(x_enc, d_enc)
>>> out['rgb'].shape, out['sigma'].shape
(torch.Size([1024, 3]), torch.Size([1024, 1]))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.encoding import PositionalEncoder


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in a model.

    Args:
        model: Any ``torch.nn.Module``.

    Returns:
        Integer count of parameters where ``requires_grad=True``.

    Example:
        >>> count_parameters(NeRF(pos_enc_dim=63))
        1193347
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# NeRF MLP
# ---------------------------------------------------------------------------

class NeRF(nn.Module):
    """NeRF MLP: maps encoded positions (and directions) to RGB + density.

    Architecture (following Mildenhall et al. 2020):

    1. **Main trunk** – ``num_layers`` fully-connected layers (default 8),
       each with ``hidden_dim`` units (default 256) and ReLU activations.
       A *skip connection* re-injects the original position encoding at
       the layer indices listed in ``skip_connect`` (default: layer 4).

    2. **Density head** – single linear layer followed by ReLU to produce
       a non-negative scalar density σ (shape ``(B, 1)``).

    3. **Colour head** – depends on ``use_viewdirs``:
       - ``True``: a feature vector from the trunk is concatenated with the
         encoded view direction, passed through one more linear layer, and
         activated with sigmoid to produce RGB in [0, 1].
       - ``False``: the feature vector is directly projected to RGB via a
         linear layer + sigmoid (no view-direction conditioning).

    Args:
        pos_enc_dim: Dimensionality of the encoded 3-D position input.
        dir_enc_dim: Dimensionality of the encoded view-direction input
            (ignored when ``use_viewdirs=False``).  Default 0.
        num_layers: Number of layers in the main trunk.  Default 8.
        hidden_dim: Width of each hidden layer.  Default 256.
        skip_connect: Layer indices at which the position encoding is
            concatenated to the hidden features.  Default ``(4,)``.
        use_viewdirs: Whether to condition colour on view direction.
            Default ``True``.

    Input shapes:
        x_encoded: ``(B, pos_enc_dim)`` – encoded 3-D positions.
        d_encoded: ``(B, dir_enc_dim)`` – encoded view directions
            (required when ``use_viewdirs=True``).

    Output:
        dict with keys:
            ``'rgb'``   : ``(B, 3)`` float32, values in [0, 1].
            ``'sigma'`` : ``(B, 1)`` float32, values >= 0.
    """

    def __init__(
        self,
        pos_enc_dim: int,
        dir_enc_dim: int = 0,
        num_layers: int = 8,
        hidden_dim: int = 256,
        skip_connect: Tuple[int, ...] = (4,),
        use_viewdirs: bool = True,
    ) -> None:
        super().__init__()

        self.pos_enc_dim = pos_enc_dim
        self.dir_enc_dim = dir_enc_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skip_connect = set(skip_connect)
        self.use_viewdirs = use_viewdirs

        # ------------------------------------------------------------------
        # Build the main trunk layers
        # ------------------------------------------------------------------
        # Layer i receives hidden_dim features; at skip layers it also
        # receives the re-injected position encoding (pos_enc_dim extra dims).
        trunk_layers: List[nn.Linear] = []
        in_dim = pos_enc_dim
        for i in range(num_layers):
            trunk_layers.append(nn.Linear(in_dim, hidden_dim))
            # Next layer's input size: at skip layers, add pos_enc_dim
            if (i + 1) in self.skip_connect:
                in_dim = hidden_dim + pos_enc_dim
            else:
                in_dim = hidden_dim
        self.trunk = nn.ModuleList(trunk_layers)

        # ------------------------------------------------------------------
        # Density head: linear -> ReLU (non-negative)
        # ------------------------------------------------------------------
        self.density_head = nn.Linear(hidden_dim, 1)

        # ------------------------------------------------------------------
        # Colour head
        # ------------------------------------------------------------------
        # First project trunk output to a feature vector (256-dim)
        self.feature_layer = nn.Linear(hidden_dim, hidden_dim)

        if use_viewdirs:
            # Concatenate feature with encoded direction, then predict RGB
            self.color_layer = nn.Linear(hidden_dim + dir_enc_dim, 3)
        else:
            # Predict RGB directly from feature vector
            self.color_layer = nn.Linear(hidden_dim, 3)

        # ------------------------------------------------------------------
        # Weight initialisation
        # ------------------------------------------------------------------
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x_encoded: torch.Tensor,
        d_encoded: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the NeRF MLP.

        Args:
            x_encoded: Sinusoidally-encoded 3-D positions,
                shape ``(B, pos_enc_dim)``.
            d_encoded: Sinusoidally-encoded view directions,
                shape ``(B, dir_enc_dim)``.  Required when
                ``use_viewdirs=True``; ignored otherwise.

        Returns:
            dict:
                ``'rgb'``   – ``(B, 3)`` colour, sigmoid-activated.
                ``'sigma'`` – ``(B, 1)`` volume density, ReLU-activated.

        Raises:
            ValueError: If ``use_viewdirs=True`` but ``d_encoded`` is None.
        """
        if self.use_viewdirs and d_encoded is None:
            raise ValueError(
                "use_viewdirs=True but d_encoded was not provided. "
                "Pass encoded view directions to forward()."
            )

        # ----- Main trunk -----
        h = x_encoded  # (B, pos_enc_dim)
        for i, layer in enumerate(self.trunk):
            h = F.relu(layer(h), inplace=True)
            # Inject skip at the *next* layer's input
            if (i + 1) in self.skip_connect:
                h = torch.cat([h, x_encoded], dim=-1)  # (B, hidden_dim + pos_enc_dim)

        # ----- Density head -----
        sigma = F.relu(self.density_head(h), inplace=False)  # (B, 1)

        # ----- Colour head -----
        feature = self.feature_layer(h)  # (B, hidden_dim)

        if self.use_viewdirs:
            # Condition colour on view direction
            feature_dir = torch.cat([feature, d_encoded], dim=-1)  # (B, hidden_dim + dir_enc_dim)
            rgb = torch.sigmoid(self.color_layer(feature_dir))      # (B, 3)
        else:
            rgb = torch.sigmoid(self.color_layer(feature))          # (B, 3)

        return {"rgb": rgb, "sigma": sigma}

    def extra_repr(self) -> str:
        return (
            f"pos_enc_dim={self.pos_enc_dim}, "
            f"dir_enc_dim={self.dir_enc_dim}, "
            f"num_layers={self.num_layers}, "
            f"hidden_dim={self.hidden_dim}, "
            f"skip_connect={sorted(self.skip_connect)}, "
            f"use_viewdirs={self.use_viewdirs}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_nerf(
    config: dict,
) -> Tuple["PositionalEncoder", Optional["PositionalEncoder"], NeRF]:
    """Construct positional encoders and a NeRF model from a config dict.

    Expected config keys (all optional – sensible defaults are used):

    ============== ============ ==========================================
    Key            Default      Description
    ============== ============ ==========================================
    pos_enc_freqs  10           Number of frequency bands for positions.
    dir_enc_freqs  4            Number of frequency bands for directions.
    num_layers     8            Number of trunk layers.
    hidden_dim     256          Width of each hidden layer.
    skip_connect   [4]          Layer indices for skip connections.
    use_viewdirs   True         Whether to use view-direction conditioning.
    ============== ============ ==========================================

    Args:
        config: Dictionary of hyper-parameters (see table above).

    Returns:
        Tuple of ``(pos_encoder, dir_encoder, model)`` where:
            - ``pos_encoder`` is a :class:`~src.encoding.PositionalEncoder`
              for 3-D positions.
            - ``dir_encoder`` is a :class:`~src.encoding.PositionalEncoder`
              for view directions, or ``None`` if ``use_viewdirs=False``.
            - ``model`` is an initialised :class:`NeRF` instance.

    Example:
        >>> pos_enc, dir_enc, nerf = create_nerf({
        ...     "pos_enc_freqs": 10,
        ...     "dir_enc_freqs": 4,
        ...     "use_viewdirs": True,
        ... })
        >>> # Encode a batch of 3-D points and directions
        >>> pts  = torch.zeros(1, 3)
        >>> dirs = torch.zeros(1, 3)
        >>> out = nerf(pos_enc(pts), dir_enc(dirs))
    """
    pos_enc_freqs: int = config.get("pos_enc_freqs", 10)
    dir_enc_freqs: int = config.get("dir_enc_freqs", 4)
    num_layers: int    = config.get("num_layers", 8)
    hidden_dim: int    = config.get("hidden_dim", 256)
    skip_connect       = tuple(config.get("skip_connect", [4]))
    use_viewdirs: bool = config.get("use_viewdirs", True)

    # --- Position encoder (input dim = 3) ---
    pos_encoder = PositionalEncoder(
        num_freqs=pos_enc_freqs,
        include_input=True,
        log_sampling=True,
    )
    # Probe output dimension with a dummy tensor
    _dummy_pos = torch.zeros(1, 3)
    pos_enc_dim: int = pos_encoder(_dummy_pos).shape[-1]

    # --- Direction encoder ---
    dir_encoder: Optional[PositionalEncoder] = None
    dir_enc_dim: int = 0
    if use_viewdirs:
        dir_encoder = PositionalEncoder(
            num_freqs=dir_enc_freqs,
            include_input=True,
            log_sampling=True,
        )
        _dummy_dir = torch.zeros(1, 3)
        dir_enc_dim = dir_encoder(_dummy_dir).shape[-1]

    # --- NeRF model ---
    model = NeRF(
        pos_enc_dim=pos_enc_dim,
        dir_enc_dim=dir_enc_dim,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        skip_connect=skip_connect,
        use_viewdirs=use_viewdirs,
    )

    return pos_encoder, dir_encoder, model
