from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional

import torch

from .core import GaussianMixture, MatrixPWCProtocol
from .gmm_spec import GMMSpec
from .metrics import (
    CostToGoResult,
    SpeciationTimingResult,
    VelocityGradientSensitivityResult,
    AutoCorrelationResult,
    cost_to_go_decomposition,
    speciation_timing,
    velocity_gradient_sensitivity,
    auto_correlation,
)
from .pid import LQGMPID, _SimResult
from .protocol_spec import AdaPIDProtocolSpec

Tensor = torch.Tensor
MetricName = Literal[
    "cost_to_go_decomposition",
    "velocity_gradient_sensitivity",
    "speciation_timing",
    "auto_correlation",
]


@dataclass(frozen=True)
class ExperimentSpec:
    """Thin experiment specification for AdaPID-revision sweeps.

    The experiment layer is intentionally restricted: it composes a GMM target,
    a restricted AdaPID protocol, a simulation budget, and a selected subset of
    QoS diagnostics into a single reproducible run.
    """

    gmm_spec: GMMSpec
    protocol_spec: AdaPIDProtocolSpec
    n_particles: int = 256
    n_steps: int = 2000
    seed: int = 0
    dtype: torch.dtype = torch.float64
    device: str | torch.device = "cpu"
    bc_eps: float = 1e-6
    metrics: tuple[MetricName, ...] = (
        "cost_to_go_decomposition",
        "velocity_gradient_sensitivity",
        "speciation_timing",
    )
    store_trajectories: bool = False
    x0: Optional[Tensor] = None
    metric_options: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.n_particles <= 0:
            raise ValueError(f"n_particles must be positive, got {self.n_particles}")
        if self.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {self.n_steps}")
        bad = [m for m in self.metrics if m not in _METRIC_DISPATCH]
        if bad:
            raise ValueError(f"unsupported metrics: {bad}")
        if self.gmm_spec.d != self.protocol_spec.d:
            raise ValueError(
                f"dimension mismatch: gmm_spec.d={self.gmm_spec.d} vs protocol_spec.d={self.protocol_spec.d}"
            )
        if self.x0 is not None and tuple(self.x0.shape) != (self.gmm_spec.d,):
            raise ValueError(f"x0 must have shape ({self.gmm_spec.d},), got {tuple(self.x0.shape)}")

    @property
    def d(self) -> int:
        return self.gmm_spec.d

    def build_target(self) -> GaussianMixture:
        return self.gmm_spec.build()

    def build_protocol(self) -> MatrixPWCProtocol:
        return self.protocol_spec.build()

    def build_x0(self) -> Tensor:
        if self.x0 is not None:
            return self.x0.to(device=self.device, dtype=self.dtype)
        return torch.zeros(self.d, device=self.device, dtype=self.dtype)

    def build_pid(self) -> LQGMPID:
        return LQGMPID(
            protocol=self.build_protocol(),
            target=self.build_target(),
            x0=self.build_x0(),
            bc_eps=float(self.bc_eps),
        )


@dataclass
class ExperimentResult:
    spec: ExperimentSpec
    pid: LQGMPID
    sim: _SimResult
    metrics: Dict[str, Any]
    summary: Dict[str, Any]
    target: GaussianMixture
    protocol: MatrixPWCProtocol
    trajectories: Optional[Tensor] = None

    def metric(self, name: MetricName) -> Any:
        return self.metrics[name]


# ---------------------------------------------------------------------------
# Public runner
# ---------------------------------------------------------------------------

def run_experiment(spec: ExperimentSpec) -> ExperimentResult:
    pid = spec.build_pid().precompute()
    sim = pid.simulate(
        B=spec.n_particles,
        n_steps=spec.n_steps,
        seed=spec.seed,
        dtype=spec.dtype,
        device=str(spec.device),
    )

    results: Dict[str, Any] = {}
    for name in spec.metrics:
        metric_fn = _METRIC_DISPATCH[name]
        kwargs = dict(spec.metric_options.get(name, {}))
        results[name] = metric_fn(
            pid,
            sim,
            B=spec.n_particles,
            n_steps=spec.n_steps,
            seed=spec.seed,
            dtype=spec.dtype,
            device=str(spec.device),
            **kwargs,
        )

    summary = summarize_experiment(spec, pid, sim, results)
    traj = sim.traj if spec.store_trajectories else None
    return ExperimentResult(
        spec=spec,
        pid=pid,
        sim=sim,
        metrics=results,
        summary=summary,
        target=pid.target,
        protocol=pid.protocol,
        trajectories=traj,
    )


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def summarize_experiment(
    spec: ExperimentSpec,
    pid: LQGMPID,
    sim: _SimResult,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "d": spec.d,
        "K": int(pid.target.weights.numel()),
        "n_particles": int(spec.n_particles),
        "n_steps": int(spec.n_steps),
        "seed": int(spec.seed),
        "beta_family": spec.protocol_spec.family,
        "n_breaks": int(pid.protocol.breaks.numel()),
        "device": str(spec.device),
    }

    ctg = metrics.get("cost_to_go_decomposition")
    if isinstance(ctg, CostToGoResult):
        summary.update({
            "final_cost_total": float(ctg.total_cum[-1].item()),
            "final_cost_kinetic": float(ctg.kinetic_cum[-1].item()),
            "final_cost_potential": float(ctg.potential_cum[-1].item()),
            "final_kinetic_share": float(ctg.kinetic_share[-1].item()),
            "final_potential_share": float(ctg.potential_share[-1].item()),
        })

    vgs = metrics.get("velocity_gradient_sensitivity")
    if isinstance(vgs, VelocityGradientSensitivityResult):
        summary.update({
            "vgs_mean_frob_sq": float(vgs.mean_frob_sq.mean().item()),
            "vgs_mean_trace": float(vgs.mean_trace.mean().item()),
            "vgs_mean_eig_max": float(vgs.mean_eig_max.mean().item()),
            "vgs_mean_eig_min": float(vgs.mean_eig_min.mean().item()),
            "vgs_particles_used": int(vgs.n_particles_used),
        })

    st = metrics.get("speciation_timing")
    if isinstance(st, SpeciationTimingResult):
        rt = st.reliable_time
        finite = torch.isfinite(rt)
        if finite.any():
            median_rt = float(rt[finite].median().item())
            mean_rt = float(rt[finite].mean().item())
        else:
            median_rt = float("inf")
            mean_rt = float("inf")
        summary.update({
            "spec_terminal_accuracy": float(st.accuracy[-1].item()),
            "spec_terminal_risk": float(st.risk[-1].item()),
            "spec_median_reliable_time": median_rt,
            "spec_mean_reliable_time": mean_rt,
            "spec_fraction_reliable": float(finite.float().mean().item()),
        })

    ac = metrics.get("auto_correlation")
    if isinstance(ac, AutoCorrelationResult):
        summary.update({
            "ac_final_A": float(ac.A[-1].item()),
            "ac_final_Ahat": float(ac.Ahat[-1].item()),
            "ac_t_cross_A_half": float(ac.t_cross_A_half),
            "ac_t_cross_Ahat_half": float(ac.t_cross_Ahat_half),
        })

    return summary


# ---------------------------------------------------------------------------
# Convenience sweep helpers
# ---------------------------------------------------------------------------

def run_experiments(specs: Iterable[ExperimentSpec]) -> list[ExperimentResult]:
    return [run_experiment(spec) for spec in specs]


def summarize_results(results: Iterable[ExperimentResult]) -> list[Dict[str, Any]]:
    return [res.summary for res in results]


_METRIC_DISPATCH = {
    "cost_to_go_decomposition": cost_to_go_decomposition,
    "velocity_gradient_sensitivity": velocity_gradient_sensitivity,
    "speciation_timing": speciation_timing,
    "auto_correlation": auto_correlation,
}
