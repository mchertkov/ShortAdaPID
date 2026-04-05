"""
core.py — Dataclasses for LQ-GM-PID.

Design principles
-----------------
* Pure PyTorch throughout (CUDA / autograd compatible).
* GaussianMixture and TimeDomain are drop-in compatible with guided_continuous
  (same field names, same validation logic).
* MatrixPWCProtocol is the *general* version of guided_continuous.PWCProtocol:
  it carries full d×d sigma and beta matrices (σ_k, β_k, ν_k) per interval.
  The special case σ_k=0, β_k=β_scalar_k * I_d recovers the scalar API exactly.
* CoeffState holds one Green-function coefficient snapshot (A, B, C, θ_x, θ_y, ζ).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# TimeDomain  (identical to guided_continuous.TimeDomain)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeDomain:
    """Clamp t to [eps, 1-eps] to avoid endpoint singularities.

    Field-for-field identical to guided_continuous.TimeDomain so the same
    instance can be shared between both packages.
    """
    eps: float = 1e-3

    def clamp(self, t: Tensor) -> Tensor:
        lo = torch.as_tensor(self.eps,       dtype=t.dtype, device=t.device)
        hi = torch.as_tensor(1.0 - self.eps, dtype=t.dtype, device=t.device)
        return torch.clamp(t, lo, hi)


# ---------------------------------------------------------------------------
# GaussianMixture  (identical to guided_continuous.GaussianMixture)
# ---------------------------------------------------------------------------

@dataclass
class GaussianMixture:
    """Gaussian mixture target / initial distribution.

    Fields
    ------
    weights : (M,)      automatically normalised
    means   : (M, d)
    covs    : (M, d, d) must be SPD

    Field-for-field compatible with guided_continuous.GaussianMixture.
    """
    weights: Tensor
    means:   Tensor
    covs:    Tensor

    def __post_init__(self):
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got {tuple(self.weights.shape)}")
        M = int(self.weights.numel())
        if self.means.shape[0] != M:
            raise ValueError("means.shape[0] must equal len(weights)")
        if (self.covs.ndim != 3
                or self.covs.shape[0] != M
                or self.covs.shape[1] != self.covs.shape[2]):
            raise ValueError("covs must be (M, d, d)")
        if self.covs.shape[1] != self.means.shape[1]:
            raise ValueError("means / covs dimension mismatch")
        self.weights = self.weights / self.weights.sum()

    @property
    def M(self) -> int:
        return int(self.weights.numel())

    @property
    def K(self) -> int:
        """Alias used by guided_continuous."""
        return self.M

    @property
    def d(self) -> int:
        return int(self.means.shape[1])

    @property
    def precisions(self) -> Tensor:
        """P_k = Σ_k^{-1},  shape (M, d, d)."""
        return torch.linalg.inv(self.covs)

    def to(self, *, device=None, dtype=None) -> "GaussianMixture":
        kw: dict = {}
        if device is not None: kw["device"] = device
        if dtype  is not None: kw["dtype"]  = dtype
        return GaussianMixture(
            weights=self.weights.to(**kw),
            means  =self.means.to(**kw),
            covs   =self.covs.to(**kw),
        )

    @staticmethod
    def single(mean: Tensor, cov: Tensor) -> "GaussianMixture":
        """Single-component convenience constructor."""
        w = torch.ones(1, device=mean.device, dtype=mean.dtype)
        return GaussianMixture(
            weights=w, means=mean.unsqueeze(0), covs=cov.unsqueeze(0)
        )


# ---------------------------------------------------------------------------
# MatrixPWCProtocol — general (d×d matrix) piecewise-constant protocol
# ---------------------------------------------------------------------------

@dataclass
class MatrixPWCProtocol:
    """Piecewise-constant protocol with full matrix coefficients.

    On interval I_k = [breaks[k], breaks[k+1]]:
        f_t(x)  =  σ_k @ x
        V_t(x)  =  ½ (x − ν_k)ᵀ β_k (x − ν_k)

    Fields
    ------
    breaks : (K+1,)    strictly increasing, breaks[0]=0, breaks[-1]=1
    sigma  : (K, d, d) drift matrices σ_k  (arbitrary square)
    beta   : (K, d, d) SPD potential matrices β_k
    nu     : (K, d)    potential centres ν_k

    Relation to guided_continuous.PWCProtocol
    -----------------------------------------
    The old API stores a scalar β_k (broadcast to β_k * I_d) and σ_k = 0.
    Use MatrixPWCProtocol.from_scalar_beta() to convert exactly.
    """
    breaks:      Tensor          # (K+1,)
    sigma:       Tensor          # (K, d, d)
    beta:        Tensor          # (K, d, d)
    nu:          Tensor          # (K, d)
    time_domain: TimeDomain = field(default_factory=TimeDomain)

    def __post_init__(self):
        b = self.breaks
        if b.ndim != 1 or b.numel() < 2:
            raise ValueError("breaks must be 1-D with ≥ 2 entries")
        if not torch.all(b[1:] > b[:-1]):
            raise ValueError("breaks must be strictly increasing")
        if abs(float(b[0].item())) > 1e-12:
            raise ValueError("breaks[0] must be 0")
        if abs(float(b[-1].item()) - 1.0) > 1e-12:
            raise ValueError("breaks[-1] must be 1")
        K, d = self.K, self.d
        for name, tensor, expected in [
            ("sigma", self.sigma, (K, d, d)),
            ("beta",  self.beta,  (K, d, d)),
            ("nu",    self.nu,    (K, d)),
        ]:
            if tuple(tensor.shape) != expected:
                raise ValueError(f"{name} must be {expected}, "
                                 f"got {tuple(tensor.shape)}")

    @property
    def K(self) -> int:
        """Number of PWC intervals."""
        return int(self.breaks.numel()) - 1

    @property
    def d(self) -> int:
        return int(self.sigma.shape[-1])

    @property
    def device(self) -> torch.device:
        return self.breaks.device

    @property
    def dtype(self) -> torch.dtype:
        return self.breaks.dtype

    def locate(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return (idx, tau, Delta) for scalar or batched t.

        tau   = t_clamped − breaks[idx]
        Delta = breaks[idx+1] − breaks[idx]

        Matches guided_continuous.PWCProtocol.locate() interface.
        """
        t   = torch.as_tensor(t, dtype=self.dtype, device=self.device)
        t_c = self.time_domain.clamp(t)
        idx = torch.searchsorted(self.breaks, t_c, right=True) - 1
        idx = torch.clamp(idx, 0, self.K - 1)
        tau   = t_c - self.breaks[idx]
        Delta = self.breaks[idx + 1] - self.breaks[idx]
        return idx, tau, Delta

    def interval_length(self, k: int) -> float:
        return float((self.breaks[k + 1] - self.breaks[k]).item())

    def to(self, *, device=None, dtype=None) -> "MatrixPWCProtocol":
        kw: dict = {}
        if device is not None: kw["device"] = device
        if dtype  is not None: kw["dtype"]  = dtype
        return MatrixPWCProtocol(
            breaks=self.breaks.to(**kw),
            sigma =self.sigma.to(**kw),
            beta  =self.beta.to(**kw),
            nu    =self.nu.to(**kw),
            time_domain=self.time_domain,
        )

    # ------------------------------------------------------------------
    # Constructors / converters
    # ------------------------------------------------------------------

    @staticmethod
    def from_scalar_beta(
        breaks:       Tensor,           # (K+1,)
        beta_scalars: Tensor,           # (K,)
        nu:           Tensor,           # (K, d)
        *,
        time_domain:  TimeDomain = TimeDomain(),
    ) -> "MatrixPWCProtocol":
        """Convert guided_continuous scalar-β protocol to matrix form.

        Sets sigma[k] = 0  and  beta[k] = beta_scalars[k] * I_d.

        This is the exact isotropic special case: the matrix-exp Hamiltonian
        decouples per-dimension into d identical 2×2 blocks whose solutions
        are the coth / csch functions in ContinuousCoeffs.
        """
        K = nu.shape[0]
        d = nu.shape[1]
        dev, dty = breaks.device, breaks.dtype
        sigma = torch.zeros(K, d, d, device=dev, dtype=dty)
        I_d   = torch.eye(d, device=dev, dtype=dty)
        beta  = beta_scalars.to(dev, dty).reshape(K, 1, 1) * I_d.unsqueeze(0)
        return MatrixPWCProtocol(
            breaks=breaks, sigma=sigma, beta=beta, nu=nu,
            time_domain=time_domain,
        )

    @staticmethod
    def trivial(K: int, d: int, *,
                device: str = "cpu",
                dtype:  torch.dtype = torch.float64) -> "MatrixPWCProtocol":
        """Zero drift, identity potential, zero centres, uniform grid."""
        breaks = torch.linspace(0.0, 1.0, K + 1, device=device, dtype=dtype)
        sigma  = torch.zeros(K, d, d, device=device, dtype=dtype)
        beta   = torch.eye(d, device=device, dtype=dtype).unsqueeze(0).expand(K, -1, -1).clone()
        nu     = torch.zeros(K, d, device=device, dtype=dtype)
        return MatrixPWCProtocol(breaks=breaks, sigma=sigma, beta=beta, nu=nu)


# ---------------------------------------------------------------------------
# CoeffState — Green-function coefficients at one time slice
# ---------------------------------------------------------------------------

@dataclass
class CoeffState:
    """Gaussian-exponential coefficients of G^(±)_t(x | y).

    Parameterisation (eq 13 of notes):
        log G(x|y) = −½ xᵀ A x  +  xᵀ B y  −  ½ yᵀ C y
                     +  θ_xᵀ x  +  θ_yᵀ y  +  ζ

    Fields
    ------
    A, C     : (d, d)  symmetric
    B        : (d, d)  general
    theta_x  : (d,)
    theta_y  : (d,)
    zeta     : scalar tensor | None  (optional; cancels in most ratios)
    """
    A:       Tensor
    B:       Tensor
    C:       Tensor
    theta_x: Tensor
    theta_y: Tensor
    zeta:    Optional[Tensor] = None

    @property
    def d(self) -> int:
        return int(self.A.shape[0])

    @property
    def device(self) -> torch.device:
        return self.A.device

    @property
    def dtype(self) -> torch.dtype:
        return self.A.dtype

    def to(self, *, device=None, dtype=None) -> "CoeffState":
        kw: dict = {}
        if device is not None: kw["device"] = device
        if dtype  is not None: kw["dtype"]  = dtype
        z = self.zeta.to(**kw) if self.zeta is not None else None
        return CoeffState(
            A=self.A.to(**kw), B=self.B.to(**kw), C=self.C.to(**kw),
            theta_x=self.theta_x.to(**kw),
            theta_y=self.theta_y.to(**kw),
            zeta=z,
        )
