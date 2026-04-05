"""lqgm_pid_ada — AdaPID-focused LQ-GM-PID optimal control for Gaussian-mixture diffusion bridges.

Public API
----------
    LQGMPID              high-level controller (precompute + control + simulate)
    MatrixPWCProtocol    piecewise-constant drift/potential schedule
    GaussianMixture      terminal target distribution
    TimeDomain           time-axis parameters (eps)
    CoeffState           raw Riccati coefficient struct (advanced use)
    backward_sweep       raw backward Riccati sweep
    forward_sweep        raw forward Riccati sweep
    gmm_control          raw batched control evaluation
    exact_marginal_gmm   exact instantaneous marginal as a Gaussian mixture
    make_standard_forward_sweep  forward sweep with standard BC (for density)
"""

from .core import (
    CoeffState,
    GaussianMixture,
    MatrixPWCProtocol,
    TimeDomain,
)
from .sweep import backward_sweep, forward_sweep
from .control import eval_bwd, eval_fwd, gmm_control
from .pid import LQGMPID
from .density import exact_marginal_gmm, make_standard_forward_sweep

from .protocol_spec import (
    AdaPIDProtocolSpec,
    make_constant_beta_spec,
    make_pwc_beta_spec,
    make_restricted_protocol,
)
from .metrics import (
    CostToGoResult,
    VelocityGradientSensitivityResult,
    AutoCorrelationResult,
    SpeciationTimingResult,
    cost_to_go_decomposition,
    velocity_gradient_sensitivity,
    auto_correlation,
    speciation_timing,
    predicted_state_and_responsibilities,
    target_posterior,
)

from .experiment import (
    ExperimentSpec,
    ExperimentResult,
    run_experiment,
    run_experiments,
    summarize_experiment,
    summarize_results,
)

from .gmm_spec import (
    GMMSpec,
    build_gmm,
    make_isotropic_codebook_gmm,
    make_diag_anisotropic_gmm,
    make_product_gmm,
    make_isotropic_codebook_spec,
    make_diag_anisotropic_spec,
    make_product_gmm_spec,
)

__all__ = [
    # High-level
    "LQGMPID",
    # Data types
    "CoeffState",
    "GaussianMixture",
    "MatrixPWCProtocol",
    "TimeDomain",
    # Sweeps
    "backward_sweep",
    "forward_sweep",
    # Control
    "eval_bwd",
    "eval_fwd",
    "gmm_control",
    # Density
    "exact_marginal_gmm",
    "make_standard_forward_sweep",
    # Restricted AdaPID experiment layer
    "AdaPIDProtocolSpec",
    "make_constant_beta_spec",
    "make_pwc_beta_spec",
    "make_restricted_protocol",
    # Experiment layer
    "ExperimentSpec",
    "ExperimentResult",
    "run_experiment",
    "run_experiments",
    "summarize_experiment",
    "summarize_results",
    # GMM benchmark layer
    "GMMSpec",
    "build_gmm",
    "make_isotropic_codebook_gmm",
    "make_diag_anisotropic_gmm",
    "make_product_gmm",
    "make_isotropic_codebook_spec",
    "make_diag_anisotropic_spec",
    "make_product_gmm_spec",
    # Metrics
    "CostToGoResult",
    "VelocityGradientSensitivityResult",
    "AutoCorrelationResult",
    "SpeciationTimingResult",
    "cost_to_go_decomposition",
    "velocity_gradient_sensitivity",
    "auto_correlation",
    "speciation_timing",
    "predicted_state_and_responsibilities",
    "target_posterior",
]


from .optimize import (
    DeterministicCurveResult,
    OptimizeBetaResult,
    deterministic_curves,
    objective_from_curves,
    optimize_pwc_beta,
    scan_constant_beta,
    make_pid_from_beta,
)
