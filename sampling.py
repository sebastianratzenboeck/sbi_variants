import torch
from typing import Optional, Sequence, Union


# ---------------------------------------------------------------------------
# Inference helpers — standalone functions (no trainer dependency)
# ---------------------------------------------------------------------------

def build_inference_edge_mask(
    batch_size: int,
    num_nodes: int,
    observed_mask: Optional[torch.Tensor] = None,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a dense edge mask for inference, with observed-mask filtering.

    At inference we always use fully connected attention (dense_ratio=1.0).
    If *observed_mask* is provided, unobserved nodes are cut off from
    attention via the outer AND product, and self-loops are restored so
    that softmax never encounters an all-masked row.

    Args:
        batch_size: Number of samples in the batch.
        num_nodes:  Number of nodes (features) per sample.
        observed_mask: Optional (B, N) or (B, N, 1) binary mask
            (1 = observed, 0 = unobserved).
        device: Target device.

    Returns:
        (B, N, N) bool tensor — True means attention is allowed.
    """
    device = torch.device(device)
    masks = torch.ones(batch_size, num_nodes, num_nodes,
                       dtype=torch.bool, device=device)

    if observed_mask is not None:
        obs = observed_mask.to(device)
        if obs.dim() == 3:
            obs = obs.squeeze(-1)
        obs = obs.bool()
        obs_edge = obs.unsqueeze(1) & obs.unsqueeze(2)     # (B, N, N)
        masks = masks & obs_edge
        # Restore self-loops so softmax always has at least one valid entry
        diag_idx = torch.arange(num_nodes, device=device)
        masks[:, diag_idx, diag_idx] = True

    return masks


def build_inference_condition_mask(
    batch_size: int,
    num_nodes: int,
    conditioned_indices: Sequence[int],
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build a deterministic condition mask for inference.

    Args:
        batch_size: Number of samples in the batch.
        num_nodes:  Total number of nodes (features).
        conditioned_indices: Column indices that are conditioned on
            (the model will keep these fixed and infer the rest).
        device: Target device.

    Returns:
        (B, num_nodes, 1) float tensor — 1.0 for conditioned dims, 0.0 elsewhere.
    """
    device = torch.device(device)
    mask = torch.zeros(batch_size, num_nodes, 1,
                       dtype=torch.float32, device=device)
    if conditioned_indices:
        idx = torch.tensor(conditioned_indices, dtype=torch.long, device=device)
        mask[:, idx, :] = 1.0
    return mask


def build_inference_node_ids(
    batch_size: int,
    num_nodes: int,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Build node ID tensor for inference.

    Returns:
        (B, num_nodes) long tensor — each row is [0, 1, ..., num_nodes-1].
    """
    device = torch.device(device)
    return torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size, -1)


# ---------------------------------------------------------------------------
# Velocity wrapper for ODE integration
# ---------------------------------------------------------------------------

class SimformerConditionalVelocity(torch.nn.Module):
    def __init__(self, model_fn, condition_mask, condition_value, node_ids, edge_mask,
                 errors=None, observed_mask=None):
        super().__init__()
        # Normalize shapes/dtypes/devices
        if condition_mask.dim() == 2:
            condition_mask = condition_mask.unsqueeze(-1)           # (B, M, 1)
        if condition_value.dim() == 2:
            condition_value = condition_value.unsqueeze(-1)         # (B, M, 1)

        device = node_ids.device
        self.model_fn = model_fn
        self.condition_mask = condition_mask.to(device=device, dtype=torch.float32)     # (B, M, 1)
        self.condition_value = condition_value.to(device=device, dtype=torch.float32)   # (B, M, 1)
        self.node_ids = node_ids.to(device)
        self.edge_mask = edge_mask.to(device)
        self.errors = errors.to(device) if errors is not None else None
        self.observed_mask = observed_mask.to(device) if observed_mask is not None else None

    def forward(self, t, x_flat):
        B, M, _ = self.condition_mask.shape
        # Accept either (B*M,) or (B*M,1); reshape to (B, M, 1)
        x = x_flat.view(B, M, 1)
        # Clamp to conditional manifold
        x = x * (1.0 - self.condition_mask) + self.condition_value * self.condition_mask
        # Time as (B,1,1)
        t_batch = torch.full((B, 1, 1), float(t), dtype=x.dtype, device=x.device)
        # Model call expects (B, M, 1) mask
        v = self.model_fn(t_batch, x, self.node_ids, self.condition_mask, self.edge_mask,
                          self.errors, observed_mask=self.observed_mask)  # (B, M, 1)
        # Zero velocity on conditioned dims.
        # Note: The Simformer model already zeros conditioned dims internally.
        # This second masking is intentional defensive redundancy — it ensures
        # correctness even if a different model is used with this wrapper.
        v = v * (1.0 - self.condition_mask)
        return v.view(-1)  # (B*M,)


# ---------------------------------------------------------------------------
# Flow sampling via Euler integration
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_batched_flow(
    model_fn,
    shape,
    condition_mask,
    condition_values,
    node_ids,
    edge_masks,
    errors=None,
    observed_mask=None,
    steps=64,
    t0=0.0,
    t1=1.0,
    device="cpu",
):
    """
    Euler integration from t0->t1.

    Args:
        model_fn:         The Simformer model (nn.Module).
        shape:            Tuple whose first element is batch size B.
        condition_mask:   (B, M) or (B, M, 1) — 1 for conditioned dims.
        condition_values: (B, M) or (B, M, 1) — values for conditioned dims.
        node_ids:         (B, M) — node ID indices.
        edge_masks:       (B, M, M) — attention mask (True = allowed).
        errors:           Optional (B, M) or (B, M, 1) — measurement errors.
        observed_mask:    Optional (B, M) or (B, M, 1) — 1 for observed dims.
        steps:            Number of Euler integration steps.
        t0:               Start time (default 0.0).
        t1:               End time (default 1.0).
        device:           Target device.

    Returns:
        (B, M, 1) tensor of samples in normalized space.
    """
    B, M = shape[0], node_ids.shape[1]
    device = torch.device(device)

    # Ensure (B,M,1) for mask/values
    if condition_mask.dim() == 2:
        condition_mask = condition_mask.unsqueeze(-1)
    if condition_values.dim() == 2:
        condition_values = condition_values.unsqueeze(-1)
    condition_mask = condition_mask.to(device=device, dtype=torch.float32)
    condition_values = condition_values.to(device=device, dtype=torch.float32)
    node_ids = node_ids.to(device)
    edge_masks = edge_masks.to(device)

    dt = (t1 - t0) / steps
    ts = torch.linspace(t0, t1, steps + 1, device=device)

    # Init state and clamp to conditional manifold
    x = torch.randn(B, M, 1, device=device)
    x = x * (1.0 - condition_mask) + condition_values * condition_mask
    x_flat = x.view(-1)  # keep 1-D throughout

    # Ensure model is in eval mode (disables dropout)
    was_training = model_fn.training
    model_fn.eval()

    velocity_fn = SimformerConditionalVelocity(
        model_fn=model_fn,
        condition_mask=condition_mask,
        condition_value=condition_values,
        node_ids=node_ids,
        edge_mask=edge_masks,
        errors=errors,
        observed_mask=observed_mask,
    )

    try:
        for t in ts[:-1]:
            dx = velocity_fn(t, x_flat)      # (B*M,)
            x_flat = x_flat + dt * dx        # Euler step
            # Clamp after each step
            x = x_flat.view(B, M, 1)
            x = x * (1.0 - condition_mask) + condition_values * condition_mask
            x_flat = x.view(-1)
    finally:
        # Restore original training mode
        if was_training:
            model_fn.train()

    # Final clamp and return (B, M, 1)
    x = x_flat.view(B, M, 1)
    x = x * (1.0 - condition_mask) + condition_values * condition_mask
    return x
