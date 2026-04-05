"""
pid.py — Top-level LQGMPID class.

Typical usage
-------------
    from lqgm_pid import LQGMPID, MatrixPWCProtocol, GaussianMixture

    proto = MatrixPWCProtocol.from_scalar_beta(breaks, beta_scalars, nu)
    target = GaussianMixture(weights, means, covs)
    x0 = torch.zeros(d)

    pid = LQGMPID(proto, target, x0)
    pid.precompute()

    # evaluate optimal control at a batch of (t, x) pairs
    u = pid.control(t, x)           # (B, d)
    u, log_psi, rho = pid.control_full(t, x)

    # integrate a trajectory
    result = pid.simulate(B=256, n_steps=2000)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from .core import CoeffState, GaussianMixture, MatrixPWCProtocol, TimeDomain
from .sweep import backward_sweep, forward_sweep
from .control import eval_bwd, eval_fwd, gmm_control

Tensor = torch.Tensor

_DEFAULT_BC_EPS = 1e-6


@dataclass
class LQGMPID:
    """LQ-GM-PID controller.

    Computes the optimal control field for a Gaussian-mixture–guided
    diffusion with piecewise-constant linear-quadratic potential.

    Parameters
    ----------
    protocol   : MatrixPWCProtocol — drift/potential schedule
    target     : GaussianMixture   — terminal target distribution
    x0         : (d,) Tensor       — deterministic starting point
    bc_eps     : precision of the δ-BC approximation (default 1e-6)

    After construction, call :meth:`precompute` (or it is called lazily on
    the first :meth:`control` call) to run the forward and backward sweeps.
    """
    protocol:  MatrixPWCProtocol
    target:    GaussianMixture
    x0:        Tensor                  # (d,)
    bc_eps:    float = _DEFAULT_BC_EPS

    # Private sweep caches
    _bwd: Optional[List[CoeffState]] = field(default=None, init=False, repr=False)
    _fwd: Optional[List[CoeffState]] = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def precompute(self) -> "LQGMPID":
        """Run backward and forward sweeps and cache the results.

        Safe to call multiple times (idempotent: only re-runs if not cached).
        Returns self for chaining.

        Note: the forward sweep uses x0_mean=None (standard two-point Green
        function).  The starting point x0 enters only through the explicit
        B^+_t x0 products in the control and density formulas.  This avoids
        double-counting x0 when x0 != 0.
        """
        if self._bwd is None:
            self._bwd = backward_sweep(self.protocol, bc_eps=self.bc_eps)
        if self._fwd is None:
            self._fwd = forward_sweep(
                self.protocol,
                bc_eps=self.bc_eps,
            )
        return self

    def reset(self) -> "LQGMPID":
        """Clear cached sweeps (e.g. after changing protocol or x0)."""
        self._bwd = None
        self._fwd = None
        return self

    @property
    def is_precomputed(self) -> bool:
        return self._bwd is not None and self._fwd is not None

    # ------------------------------------------------------------------
    # Control evaluation
    # ------------------------------------------------------------------

    def control(
        self,
        t: float | Tensor,
        x: Tensor,           # (B, d)
    ) -> Tensor:             # (B, d)
        """Evaluate the optimal control field u*(t, x).

        Lazily precomputes sweeps on first call.

        Parameters
        ----------
        t : scalar query time
        x : (B, d) batch of query positions

        Returns
        -------
        u_star : (B, d)
        """
        self.precompute()
        u_star, _, _ = gmm_control(
            t, x, self.x0,
            self._bwd, self._fwd,
            self.target, self.protocol,
        )
        return u_star

    def control_full(
        self,
        t: float | Tensor,
        x: Tensor,           # (B, d)
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Evaluate u*(t,x), log ψ_t(x), and responsibilities ρ(t,x).

        Returns
        -------
        u_star  : (B, d)
        log_psi : (B,)
        rho     : (B, M)   per-component responsibilities (sum to 1)
        """
        self.precompute()
        return gmm_control(
            t, x, self.x0,
            self._bwd, self._fwd,
            self.target, self.protocol,
        )

    def log_psi(
        self,
        t: float | Tensor,
        x: Tensor,
    ) -> Tensor:
        """Return log ψ_t(x) only (no control field)."""
        _, lp, _ = self.control_full(t, x)
        return lp

    # ------------------------------------------------------------------
    # Coefficient inspection
    # ------------------------------------------------------------------

    def bwd_at(self, t: float | Tensor) -> CoeffState:
        """Backward Green-function coefficients at time t."""
        self.precompute()
        return eval_bwd(t, self._bwd, self.protocol)

    def fwd_at(self, t: float | Tensor) -> CoeffState:
        """Forward Green-function coefficients at time t."""
        self.precompute()
        return eval_fwd(t, self._fwd, self.protocol)

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        *,
        B:       int  = 256,
        n_steps: int  = 2000,
        seed:    int  = 0,
        dtype:   torch.dtype = torch.float64,
        device:  str  = "cpu",
    ):
        """Simulate dX_t = u*(t, X_t) dt + dW_t via Euler–Maruyama.

        Returns an object with fields:
            .times  : (T,) time grid
            .traj   : (T, B, d) trajectory
            .dt     : (T-1,) step sizes

        Requires the ``guided_continuous`` SDE utilities; if not available,
        falls back to a minimal built-in Euler–Maruyama loop.
        """
        self.precompute()

        dev = torch.device(device)
        gen = torch.Generator(device=dev)
        gen.manual_seed(seed)

        td  = self.protocol.time_domain
        eps = float(td.eps)
        d   = self.protocol.d

        # breakpoint-aligned time grid
        bks = self.protocol.breaks.to(dev, dtype)
        times = _break_aligned_grid(bks, n_steps, eps, dtype, dev)
        dt_vec = times[1:] - times[:-1]

        x0_dev = self.x0.to(dev, dtype)
        x = x0_dev.unsqueeze(0).expand(B, -1).clone()   # (B, d)

        traj = torch.empty(times.numel(), B, d, dtype=dtype, device=dev)
        traj[0] = x

        for i in range(times.numel() - 1):
            t  = times[i]
            dt = dt_vec[i]
            drift = self.control(t, x)                      # (B, d)
            noise = torch.randn(B, d, dtype=dtype, device=dev, generator=gen)
            x = x + drift * dt + torch.sqrt(dt) * noise
            traj[i + 1] = x

        return _SimResult(times=times, traj=traj, dt=dt_vec)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@dataclass
class _SimResult:
    times: Tensor   # (T,)
    traj:  Tensor   # (T, B, d)
    dt:    Tensor   # (T-1,)


def _break_aligned_grid(
    breaks: Tensor,
    n_steps: int,
    eps: float,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Build a time grid on [eps, 1-eps] that includes all breakpoints."""
    lo = torch.tensor(eps,       dtype=dtype, device=device)
    hi = torch.tensor(1.0 - eps, dtype=dtype, device=device)

    # Clamp breaks to [lo, hi] and add endpoints
    b = torch.unique(torch.sort(torch.clamp(breaks, lo, hi)).values)
    if float(b[0].item()) > float(lo.item()) + 1e-15:
        b = torch.cat([lo.unsqueeze(0), b])
    if float(b[-1].item()) < float(hi.item()) - 1e-15:
        b = torch.cat([b, hi.unsqueeze(0)])

    lengths = (b[1:] - b[:-1]).clamp_min(0.0)
    L = float(lengths.sum().item())
    if L <= 0.0:
        return torch.linspace(float(lo), float(hi), n_steps + 1, dtype=dtype, device=device)

    # Allocate steps per interval proportional to length (minimum 1)
    raw = lengths / L * float(n_steps)
    n   = torch.clamp(torch.floor(raw).long(), min=1)

    deficit = n_steps - int(n.sum().item())
    if deficit > 0:
        frac = raw - torch.floor(raw)
        for idx in torch.argsort(frac, descending=True)[:deficit]:
            n[idx] += 1
    elif deficit < 0:
        for idx in torch.argsort(n, descending=True):
            if n[idx] > 1 and deficit < 0:
                n[idx] -= 1; deficit += 1

    # Build grid
    pieces = []
    for k in range(b.numel() - 1):
        seg = torch.linspace(float(b[k].item()), float(b[k+1].item()),
                             int(n[k].item()) + 1, dtype=dtype, device=device)
        pieces.append(seg[:-1] if k < b.numel() - 2 else seg)
    return torch.cat(pieces)
