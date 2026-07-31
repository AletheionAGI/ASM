from __future__ import annotations

import torch
from torch import nn

from .config import DRMConfig
from .direction_field import DirectionField
from .dynamics import DRMFlow
from .metric import RelationalMetric


class DRMStateInitializer(nn.Module):
    def __init__(self, config: DRMConfig):
        super().__init__()
        self.z0 = nn.Parameter(torch.zeros(config.d_state))
        nn.init.normal_(self.z0, std=0.02)

    def forward(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.z0.unsqueeze(0).expand(batch_size, -1).to(device)


class CausalLocalMixer(nn.Module):
    """Cheap local causal state correction for testing the Anderson-as-mixer hypothesis."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        hidden = config.directional_local_mixer_hidden_size
        in_dim = config.d_state * 3 + config.d_token + 2
        self.input_proj = nn.Linear(in_dim, hidden)
        self.depthwise = nn.ModuleList(
            [
                nn.Conv1d(
                    hidden,
                    hidden,
                    kernel_size=config.directional_local_mixer_kernel_size,
                    dilation=config.directional_local_mixer_dilation_growth**layer,
                    groups=hidden,
                    bias=True,
                )
                for layer in range(config.directional_local_mixer_layers)
            ]
        )
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(hidden, hidden, kernel_size=1, bias=True) for _ in range(config.directional_local_mixer_layers)]
        )
        self.output_proj = nn.Linear(hidden, config.d_state)

    def forward(
        self,
        z_start: torch.Tensor,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        local_delta: torch.Tensor,
        dim: torch.Tensor,
        risk_mass: torch.Tensor,
    ) -> torch.Tensor:
        previous = torch.cat([z_start.unsqueeze(1), states[:, :-1]], dim=1)
        residual = states - previous
        features = torch.cat(
            [
                states,
                residual,
                local_delta,
                token_embeddings,
                dim.unsqueeze(-1),
                risk_mass.unsqueeze(-1),
            ],
            dim=-1,
        )
        hidden = torch.nn.functional.silu(self.input_proj(features)).transpose(1, 2)
        kernel = int(self.config.directional_local_mixer_kernel_size)
        for depthwise, pointwise in zip(self.depthwise, self.pointwise):
            left_padding = depthwise.dilation[0] * (kernel - 1)
            padded = torch.nn.functional.pad(hidden, (left_padding, 0))
            mixed = pointwise(torch.nn.functional.silu(depthwise(padded)))
            hidden = hidden + mixed
        correction = self.output_proj(hidden.transpose(1, 2))
        return states + float(self.config.directional_local_mixer_scale) * correction


class TokenStateResidual(nn.Module):
    """Direct causal token-to-state path used by the D+ ablations."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.scale = float(config.token_state_residual_scale)
        self.projection = nn.Linear(config.d_token, config.d_state, bias=False)

    def forward(self, states: torch.Tensor, token_embeddings: torch.Tensor) -> torch.Tensor:
        return states + self.scale * self.projection(token_embeddings)


class SelectiveStateMemory(nn.Module):
    """Content-dependent forget/write memory with a parallel affine scan."""

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        hidden = config.selective_memory_hidden_size
        self.input_proj = nn.Linear(config.d_state + config.d_token, hidden)
        self.forget_head = nn.Linear(hidden, config.d_state)
        self.write_head = nn.Linear(hidden, config.d_state)
        self.candidate_head = nn.Linear(hidden, config.d_state)

    @staticmethod
    def _affine_scan(
        forget: torch.Tensor,
        update: torch.Tensor,
        initial: torch.Tensor,
    ) -> torch.Tensor:
        """Inclusive scan of m_t = forget_t * m_(t-1) + update_t.

        This uses associative affine-transform composition. Unlike the
        cumprod/division identity, it never divides by a vanishing prefix
        product and remains finite on long sequences.
        """
        scan_forget = forget
        scan_update = update
        offset = 1
        seq_len = forget.shape[1]
        while offset < seq_len:
            next_forget = scan_forget.clone()
            next_update = scan_update.clone()
            next_forget[:, offset:] = (
                scan_forget[:, offset:] * scan_forget[:, :-offset]
            )
            next_update[:, offset:] = (
                scan_update[:, offset:]
                + scan_forget[:, offset:] * scan_update[:, :-offset]
            )
            scan_forget = next_forget
            scan_update = next_update
            offset *= 2
        return scan_forget * initial.unsqueeze(1) + scan_update

    def forward(
        self,
        z_start: torch.Tensor,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        previous = torch.cat([z_start.unsqueeze(1), states[:, :-1]], dim=1)
        hidden = torch.nn.functional.silu(
            self.input_proj(torch.cat([previous, token_embeddings], dim=-1))
        )
        forget = torch.sigmoid(
            self.forget_head(hidden) + float(self.config.selective_memory_forget_bias)
        )
        write = torch.sigmoid(self.write_head(hidden))
        candidate = torch.tanh(self.candidate_head(hidden))

        # m_t = forget_t * m_{t-1} + write_t * candidate_t.
        scan_dtype = (
            torch.float32
            if states.dtype in {torch.float16, torch.bfloat16}
            else states.dtype
        )
        forget_scan = forget.to(scan_dtype)
        update_scan = (write * candidate).to(scan_dtype)
        memory = self._affine_scan(
            forget_scan,
            update_scan,
            z_start.to(scan_dtype),
        )
        correction = memory.to(states.dtype)
        return states + float(self.config.selective_memory_scale) * correction


class DRMRefinementLayer(nn.Module):
    """A causal state-dependent second DRM stage.

    Geometry at position t is conditioned on the preceding state produced by
    the previous stage. Positions remain parallel within the refinement stage,
    while order information is inherited from the causal prefix states.
    """

    def __init__(self, config: DRMConfig):
        super().__init__()
        self.config = config
        self.direction_field = DirectionField(config)
        self.metric = RelationalMetric(config)
        self.flow = DRMFlow(config)

    def forward(
        self,
        z_start: torch.Tensor,
        states: torch.Tensor,
        token_embeddings: torch.Tensor,
        naturalization_strength: float,
    ) -> torch.Tensor:
        batch, seq_len, d_state = states.shape
        previous = torch.cat([z_start.unsqueeze(1), states[:, :-1]], dim=1)
        flat_previous = previous.reshape(batch * seq_len, d_state)
        flat_tokens = token_embeddings.reshape(batch * seq_len, token_embeddings.shape[-1])
        directions, gates = self.direction_field(flat_previous)
        metric_diag, metric_u = self.metric(flat_previous)
        dz_raw, _ = self.flow(flat_previous, flat_tokens, directions, gates)
        dz = self.metric.naturalize(
            dz_raw,
            metric_diag,
            metric_u,
            strength=naturalization_strength,
            damping=self.config.metric_damping,
        )
        correction = (
            float(self.config.directional_refinement_scale)
            * self.config.dt
            * dz.reshape(batch, seq_len, d_state)
        )
        return states + correction
