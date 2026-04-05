"""protocol_spec.py — restricted AdaPID experiment protocol builders.

This module adds a thin experiment-facing specification layer on top of the
more general :class:`MatrixPWCProtocol` core.  The intent is to keep the LQ-GM-
PID machinery fully general while making it easy to instantiate the *restricted*
AdaPID family used in the revision experiments:

    * nu_t     = 0
    * f_t      = 0   (equivalently sigma_t = 0)
    * kappa_t  = 1   (handled by the existing core dynamics / simulator)
    * beta_t   = scalar on each interval, broadcast to beta_t I_d

The resulting objects are plain ``MatrixPWCProtocol`` instances, so all current
solvers, controls, and density routines work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import torch

from .core import MatrixPWCProtocol, TimeDomain

Tensor = torch.Tensor
BetaFamily = Literal["constant", "pwc", "optimized_pwc"]


@dataclass(frozen=True)
class AdaPIDProtocolSpec:
    """Experiment-facing specification for restricted AdaPID protocols.

    Parameters
    ----------
    d:
        Ambient dimension.
    breaks:
        1-D tensor of shape ``(K+1,)`` with ``breaks[0]=0`` and
        ``breaks[-1]=1``.
    family:
        Label used by experiment code.  In this first phase, all families map
        to explicit scalar-beta schedules.  The distinction is descriptive and
        useful for downstream bookkeeping.
    beta_values:
        1-D tensor of shape ``(K,)``.  These are the scalar stiffness values
        used on each interval and are broadcast to ``beta_k I_d``.
    time_domain:
        Endpoint clamp used by the core protocol.

    Notes
    -----
    This spec intentionally enforces the restricted AdaPID regime used in the
    revised experiments.  More general schedules (matrix beta, nonzero nu,
    nonzero sigma) should continue to be built directly via
    :class:`MatrixPWCProtocol`.
    """

    d: int
    breaks: Tensor
    family: BetaFamily
    beta_values: Tensor
    time_domain: TimeDomain = TimeDomain()

    def __post_init__(self) -> None:
        if self.d <= 0:
            raise ValueError(f"d must be positive, got {self.d}")
        if self.breaks.ndim != 1 or self.breaks.numel() < 2:
            raise ValueError("breaks must be 1-D with at least two entries")
        if self.beta_values.ndim != 1:
            raise ValueError("beta_values must be 1-D")
        K = self.breaks.numel() - 1
        if self.beta_values.numel() != K:
            raise ValueError(
                f"beta_values must have length K={K}, got {self.beta_values.numel()}"
            )
        if not torch.all(self.beta_values > 0):
            raise ValueError("all beta_values must be strictly positive")

    @property
    def K(self) -> int:
        return int(self.breaks.numel()) - 1

    @property
    def device(self) -> torch.device:
        return self.breaks.device

    @property
    def dtype(self) -> torch.dtype:
        return self.breaks.dtype

    def build(self) -> MatrixPWCProtocol:
        """Build the restricted ``MatrixPWCProtocol``.

        Returns
        -------
        MatrixPWCProtocol
            With ``sigma_k = 0``, ``nu_k = 0``, and ``beta_k = beta_values[k] I_d``.
        """
        nu = torch.zeros(self.K, self.d, device=self.device, dtype=self.dtype)
        return MatrixPWCProtocol.from_scalar_beta(
            breaks=self.breaks,
            beta_scalars=self.beta_values,
            nu=nu,
            time_domain=self.time_domain,
        )


# ---------------------------------------------------------------------------
# Convenience builders
# ---------------------------------------------------------------------------

def make_constant_beta_spec(
    *,
    d: int,
    beta: float,
    K: int = 1,
    breaks: Optional[Tensor] = None,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    time_domain: TimeDomain = TimeDomain(),
) -> AdaPIDProtocolSpec:
    """Construct a restricted constant-beta AdaPID protocol spec.

    If ``K > 1``, the same scalar beta is repeated across all intervals, which is
    useful when matching the discretization grid used by a PWC comparison.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    if breaks is None:
        breaks = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype)
    else:
        breaks = torch.as_tensor(breaks, device=device, dtype=dtype)
        K = int(breaks.numel()) - 1
    beta_values = torch.full((K,), float(beta), device=breaks.device, dtype=breaks.dtype)
    return AdaPIDProtocolSpec(
        d=d,
        breaks=breaks,
        family="constant",
        beta_values=beta_values,
        time_domain=time_domain,
    )


def make_pwc_beta_spec(
    *,
    d: int,
    breaks: Tensor | Iterable[float],
    beta_values: Tensor | Iterable[float],
    family: BetaFamily = "pwc",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float64,
    time_domain: TimeDomain = TimeDomain(),
) -> AdaPIDProtocolSpec:
    """Construct a restricted PWC AdaPID protocol spec.

    Parameters
    ----------
    family:
        Either ``"pwc"`` or ``"optimized_pwc"``.  The current builder treats
        both identically at the protocol level; the distinction is preserved for
        experiment bookkeeping.
    """
    if family not in {"pwc", "optimized_pwc"}:
        raise ValueError("family must be 'pwc' or 'optimized_pwc'")
    breaks_t = torch.as_tensor(list(breaks), device=device, dtype=dtype)
    beta_t = torch.as_tensor(list(beta_values), device=device, dtype=dtype)
    return AdaPIDProtocolSpec(
        d=d,
        breaks=breaks_t,
        family=family,
        beta_values=beta_t,
        time_domain=time_domain,
    )


def make_restricted_protocol(spec: AdaPIDProtocolSpec) -> MatrixPWCProtocol:
    """Alias for ``spec.build()`` for a functional experiment style."""
    return spec.build()
