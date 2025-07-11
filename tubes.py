"""tubes
======

A collection of utility functions for computing curvature and optimal radius
in bilayer and multilayer plate systems.  The implementation follows the
formulations presented in Bernd Schmidt's work on the spontaneous curvature of
bilayers.

Functions
---------
- :func:`calculate_Q_2_isotropic` – Isotropic quadratic form *Q₂* for a single
  material.
- :func:`calculate_Q_2_bar` – Effective quadratic form *Q̄₂* for a bilayer.
- :func:`calculate_Q_2_bar_multilayer` – *Q̄₂* for an arbitrary multilayer
  stack.
- :func:`calculate_b1_b2_multilayer` – Layer-mismatch vectors *b₁* and *b₂*.
- :func:`optimal_radius` – Optimal curvature for a bilayer film.
- :func:`radius_timoshenko` – Curvature via Timoshenko's classical formula.
- :func:`optimal_radius_multilayer` – Optimal curvature for a multilayer film.


The public interface is re‑exported via :data:`__all__` for easy wildcard
imports.
"""

from __future__ import annotations

# Standard library imports
import logging

# Third‑party imports
import numpy as np

# ---------------------------------------------------------------------------
# Public symbols
# ---------------------------------------------------------------------------
__all__: tuple[str, ...] = (
    "calculate_Q_2_isotropic",
    "calculate_Q_2_bar",
    "calculate_Q_2_bar_multilayer",
    "calculate_b1_b2_multilayer",
    "optimal_radius",
    "radius_classical_timoshenko",
    "radius_strain_theory",
    "optimal_radius_multilayer",
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())  # Silences the "No handler" warning

# ---------------------------------------------------------------------------
# Elastic‑energy helpers
# ---------------------------------------------------------------------------


def calculate_Q_2_isotropic(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """Return the *3 × 3* matrix of the bilinear form :math:`Q_2`.

    The linearised strain energy density around the identity deformation for an
    **isotropic** elastic material is characterised by the quadratic form

    .. math::
        Q_2(G) = 2\mu \|\operatorname{sym}G\|_F^2
        + \frac{2\mu\lambda}{2\mu+\lambda}\bigl(\operatorname{tr}G\bigr)^2.

    Here ``G`` is a small in‑plane deformation gradient, ``λ`` and ``μ`` are
    the Lamé parameters, ``sym`` denotes the symmetric part, and
    ``‖·‖_F`` is the Frobenius norm.

    This routine evaluates :math:`Q_2` on a fixed orthonormal basis of
    symmetric 2‑tensors and returns the associated matrix representation.

    Parameters
    ----------
    youngs_modulus
        Young's modulus :math:`E` of the material.
    poisson_ratio
        Poisson ratio :math:`\nu` of the material.

    Returns
    -------
    numpy.ndarray
        A square matrix of shape ``(3, 3)`` representing :math:`Q_2` in the
        chosen basis ``(e₁, e₂, e₃)``.
    """
    logger.debug(f"youngs_modulus={youngs_modulus}, poisson_ratio={poisson_ratio}")

    # Lamé parameters
    lambd = (
        youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    mu = youngs_modulus / (2 * (1 + poisson_ratio))

    logger.debug(f"lambda = {lambd}, mu={mu}")

    # calculate Q2, which is the linearized strain at the identity.
    def _Q2(G):
        return (
            2 * mu * np.linalg.norm(((G + np.transpose(G)) / 2), ord="fro") ** 2
            + (2 * mu * lambd) / (2 * mu + lambd) * (np.trace(G)) ** 2
        )

    # define the basis vectors
    e1 = np.array([[1, 0], [0, 0]])
    e2 = np.array([[0, 0], [0, 1]])
    e3 = np.array([[0, 1], [1, 0]])

    # define the return value
    M = np.zeros((3, 3))
    M[0, 0] = _Q2(e1)
    M[1, 1] = _Q2(e2)
    M[2, 2] = _Q2(e3)
    M[0, 1] = 0.5 * (_Q2(e1 + e2) - _Q2(e1) - _Q2(e2))
    M[1, 0] = M[0, 1]
    M[0, 2] = 0.5 * (_Q2(e1 + e3) - _Q2(e1) - _Q2(e3))
    M[2, 0] = M[0, 2]
    M[1, 2] = 0.5 * (_Q2(e2 + e3) - _Q2(e2) - _Q2(e3))
    M[2, 1] = M[1, 2]

    logger.debug(f"The matrix M is given by {M}")
    return M


def calculate_Q_2_bar(
    tau: float,
    M_top: np.ndarray,
    M_bot: np.ndarray,
) -> np.ndarray:
    """Return the tuple *(M₀, M₁, M₂, M₃)* for a bilayer.

    The effective quadratic form :math:`\bar{Q}_2` of a two‑layer plate depends
    on the relative interface position ``tau``:

    .. code:: text

               top (τ, 0.5)
        ────────────────────────
               bottom (‑0.5, τ)

    Following the notation of Bernd Schmidt (2007), the matrices ``M₀`` –
    ``M₃`` are defined by explicit layer‑averaged integrals of the single‑layer
    forms *M_top* and *M_bot*.

    Parameters
    ----------
    tau
        Signed interface position ``τ`` in the reference interval ``(‑0.5, 0.5)``.
    M_top, M_bot
        *3 × 3* matrices returned by :func:`calculate_Q_2_isotropic` for the
        top and bottom materials, respectively.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The matrices ``M₀``, ``M₁``, ``M₂``, and ``M₃``.
    """
    M1 = (tau + 1 / 2) * M_bot + (1 / 2 - tau) * M_top
    M2 = (0.5 * tau**2 - 0.5 * 1 / 2**2) * M_bot + (
        0.5 * 1 / 2**2 - 0.5 * tau**2
    ) * M_top
    M3 = (1 / 3 * tau**3 + 1 / 3 * 1 / 2**3) * M_bot + (
        1 / 3 * 1 / 2**3 - 1 / 3 * tau**3
    ) * M_top
    M0 = M3 - (M2 @ np.linalg.inv(M1)) @ M2
    return M0, M1, M2, M3


def calculate_Q_2_bar_multilayer(
    n_layers: int, interfaces: tuple, materials: tuple[np.ndarray]
) -> np.ndarray:
    """Generalise :func:`calculate_Q_2_bar` to *n* layers.

    The reference thickness is scaled to the interval ``(‑0.5, 0.5)`` with
    *n‑1* interface positions given in *ascending* order via ``interfaces``.

    Parameters
    ----------
    n_layers
        Total number of layers ``n``.
    interfaces
        Tuple of length ``n − 1`` with strictly increasing values inside the
        open interval ``(‑0.5, 0.5)``.
    materials
        Tuple of length ``n`` holding a *3 × 3* matrix for each layer as
        returned by :func:`calculate_Q_2_isotropic`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The matrices ``M₀``, ``M₁``, ``M₂``, and ``M₃`` for the multilayer.
    """
    assert len(interfaces) == n_layers - 1
    assert len(materials) == n_layers

    interfaces = (-0.5,) + interfaces + (0.5,)

    M1 = sum(
        M * (i_max - i_min)
        for (i_min, i_max), M in zip(zip(interfaces[:-1], interfaces[1:]), materials)
    )

    M2 = sum(
        M * (0.5 * i_max**2 - 0.5 * i_min**2)
        for (i_min, i_max), M in zip(zip(interfaces[:-1], interfaces[1:]), materials)
    )

    M3 = sum(
        M * (i_max**3 - i_min**3) / 3
        for (i_min, i_max), M in zip(zip(interfaces[:-1], interfaces[1:]), materials)
    )

    M0 = M3 - (M2 @ np.linalg.inv(M1)) @ M2

    return M0, M1, M2, M3


def calculate_b1_b2_multilayer(
    n_layers: int, interfaces: tuple, b: tuple, materials: tuple[np.ndarray]
):
    """Compute the mismatch vectors *b₁* and *b₂* for a multilayer stack.

    The symbol ``b`` represents the discrete lattice‑mismatch integer for each
    layer (see Schmidt, *ibid.*).  Each layer contributes a rank‑two mismatch
    tensor ``B`` with identical entries in the diagonal and zero otherwise.

    Parameters
    ----------
    n_layers
        Number of layers in the stack.
    interfaces, b, materials
        Analogous to :func:`calculate_Q_2_bar_multilayer`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The vectors ``b₁`` and ``b₂``.
    """
    assert len(interfaces) == n_layers - 1
    assert len(materials) == n_layers
    assert len(b) == n_layers

    interfaces = (-0.5,) + interfaces + (0.5,)

    # Layer Mismatch
    B = tuple([bee, bee, 0] for bee in b)

    b1 = sum(
        (i_max - i_min) * M @ bee
        for (i_min, i_max), M, bee in zip(
            zip(interfaces[:-1], interfaces[1:]), materials, B
        )
    )

    b2 = sum(
        0.5 * (i_max**2 - i_min**2) * M @ bee
        for (i_min, i_max), M, bee in zip(
            zip(interfaces[:-1], interfaces[1:]), materials, B
        )
    )

    return b1, b2


def optimal_radius(
    tau: float,
    h: float,
    total_thickness: float,
    btop: float,
    bbot: float,
    youngs_modulus: float | list,
    poisson_ratio: float | list,
    measured_radius: float = 0,
) -> dict[str, any]:
    """Compute the energetically optimal tube radius for a *bilayer* config.

    The algorithm follows Section 5 of Bernd Schmidt (2007).  The reference
    reference config. is perturbed such that the first‑order mismatch vector *b₁*
    vanishes.  The remaining quadratic optimisation reduces to a scalar
    problem in the curvature ``κ``.

    Parameters
    ----------
    tau, h, total_thickness, btop, bbot
        Geometric and mismatch parameters of the bilayer.
    youngs_modulus, poisson_ratio
        Either scalars (identical materials) or length‑two sequences
        ``(bottom, top)``.
    measured_radius
        Experimental radius :math:`R\_{\text{meas}}` for comparison.

    Returns
    -------
    dict[str, Any]
        Keys: ``kappa`` (optimal curvature), ``calculated_radius``,
        ``relative_deviation`` (``inf`` if *measured_radius* is zero),
        and the matrix ``M0``.
    """

    logger.debug("=== Running optimal radius ===")
    logger.debug(
        f"Input parameters: tau={tau}, h={h}, d={total_thickness}, "
        f"btop={btop}, bbot={bbot}"
    )

    if isinstance(youngs_modulus, list) and isinstance(poisson_ratio, list):
        M_bot = calculate_Q_2_isotropic(youngs_modulus[0], poisson_ratio[0])
        M_top = calculate_Q_2_isotropic(youngs_modulus[1], poisson_ratio[1])
    else:
        M_bot = calculate_Q_2_isotropic(youngs_modulus, poisson_ratio)
        M_top = calculate_Q_2_isotropic(youngs_modulus, poisson_ratio)

    logger.debug(f"M_bot = {M_bot}")
    logger.debug(f"M_top = {M_top}")

    M0, M1, M2, _ = calculate_Q_2_bar(
        tau=tau,
        M_bot=M_bot,
        M_top=M_top,
    )

    logger.debug(f"M0 = {M0}")
    logger.debug(f"M1 = {M1}")
    logger.debug(f"M2 = {M2}")

    # logger.debug(f"The Matrix representation of Q* is: \n {M0}")

    # Layer Mismatch
    B_bot = bbot * np.eye(3)
    B_top = btop * np.eye(3)

    B_bot_hat = B_bot[0:2, 0:2]
    B_top_hat = B_top[0:2, 0:2]

    B_hat_sym_bot = 0.5 * (B_bot_hat + np.transpose(B_bot_hat))
    B_hat_sym_top = 0.5 * (B_top_hat + np.transpose(B_top_hat))

    b_bot_vec = np.array(
        [B_hat_sym_bot[0, 0], B_hat_sym_bot[1, 1], B_hat_sym_bot[0, 1]]
    )
    b_top_vec = np.array(
        [B_hat_sym_top[0, 0], B_hat_sym_top[1, 1], B_hat_sym_top[0, 1]]
    )

    # b1 = (tau + 1 / 2) * M_bot @ b_bot_vec + (1 / 2 - tau) * M_top @ b_top_vec

    # it may be assumed that b1 = 0 by perturbing the reference configuration
    b1 = np.array([0, 0, 0])

    logger.debug(f"The value of b1 is: {b1}")

    # calculate b2
    b2 = (0.5 * tau**2 - 0.5 * 1 / 2**2) * M_bot @ b_bot_vec + (
        0.5 * 1 / 2**2 - 0.5 * tau**2
    ) * M_top @ b_top_vec

    logger.debug(f"The vector b2 is: \n {b2}.")

    f0 = np.linalg.inv(M0) @ ((M2 @ np.linalg.inv(M1) @ b1) - b2)

    # F0 = np.array([[f0[0], f0[2]], [f0[2], f0[1]]])

    logger.debug(f"The vector f0 is given by: \n {f0}")

    # we need to minimize (F - F0)^T * M0 * (F - F0) where F = kappa * (1, 0, 0)
    # This leads to minimizing the following
    # function(kappa) = alpha * kappa ^ 2 - 2 * kappa * beta
    # where alpha and beta are given by:

    alpha = np.transpose(np.array([1, 0, 0])) @ M0 @ np.array([1, 0, 0])
    beta = np.transpose(np.array([1, 0, 0])) @ M0 @ f0

    logger.debug(f"beta = {beta}, alpha = {alpha}")

    # now calculating a stationary point of function, we get kappa = beta / alpha:
    kappa = beta / alpha

    logger.debug(f"kappa = {kappa}")

    calculated_radius = total_thickness / abs(kappa) / h

    logger.debug(f"calc_rad = {calculated_radius}")

    logger.debug("=== End of run ===")

    return {
        "kappa": kappa,
        "calculated_radius": calculated_radius,
        "relative_deviation": (
            (calculated_radius - measured_radius) / measured_radius
            if measured_radius
            else np.inf
        ),
        "M0": M0,
    }


def radius_classical_timoshenko(
    E_film: float,
    E_substrate: float,
    t_film: float,
    t_substrate: float,
    lattice_film: float,
    lattice_substrate: float,
    relaxation: float = 1.0,
) -> float:
    """
    Calculates curvature kappa using Timoshenko formula.

    Parameters
    ----------
        E_top: Young's modulus of top (strained) layer
        E_bot: Young's modulus of bottom (substrate) layer
        t_top: thickness of top layer (film)
        t_bot: thickness of bottom layer (substrate)
        lattice_top: lattice constant of top layer
        lattice_bot: lattice constant of bottom layer
        relaxation: fraction of strain retained (1 = full, 0 = none)

    Returns
    -------
        kappa: curvature = (1 / radius)
    """
    alpha = E_film / E_substrate
    beta = t_film / t_substrate
    t_total = t_film + t_substrate
    strain = relaxation * (lattice_substrate - lattice_film) / lattice_film

    gamma = (1 + beta) ** 1 / (
        1
        + 4 * alpha * beta
        + 6 * alpha * beta**2
        + 4 * alpha * beta**3
        + alpha**2 * beta**4
    )

    kappa = 6 * E_film * strain * t_film / (E_substrate * t_total**2) * gamma
    return kappa


def radius_strain_theory(
    E_top: float,
    E_bot: float,
    t_top: float,
    t_bot: float,
    lattice_top: float,
    lattice_bot: float,
    poisson_ratio: float,
    relaxation: float = 1.0,
) -> float:
    """Calculate curvature kappa using continuous strain theory for a bilayer."""
    phi = E_top / E_bot
    eps = relaxation * (lattice_top / lattice_bot - 1)
    assert eps > 0, "Strain must be positive"

    r = (
        t_bot**4
        + 4 * phi * t_bot**3 * t_top
        + 6 * phi * t_bot**2 * t_top**2
        + 4 * phi * t_bot * t_top**3
        + phi**2 * t_top**4
    ) / (6 * eps * phi * (1 + poisson_ratio) * t_top * t_bot * (t_top + t_bot))

    return r


def optimal_radius_multilayer(
    h: float,
    total_thickness: float,
    n_layers: int,
    interfaces: tuple[float],
    b: tuple[float],
    youngs_moduli: tuple[float],
    poisson_ratios: tuple[float],
    measured_radius: float = 0,
) -> dict[str, any]:
    """Extend :func:`optimal_radius` to an *n*‑layer configuration."""
    logger.debug("=== Running optimal radius ===")

    materials = tuple(
        calculate_Q_2_isotropic(E, nu) for E, nu in zip(youngs_moduli, poisson_ratios)
    )

    M0, M1, M2, _ = calculate_Q_2_bar_multilayer(n_layers, interfaces, materials)

    logger.debug(f"M0 = {M0}")
    logger.debug(f"M1 = {M1}")
    logger.debug(f"M2 = {M2}")

    # logger.debug(f"The Matrix representation of Q* is: \n {M0}")

    # it may be assumed that b1 = 0 by perturbing the reference configuration
    b1 = np.array([0, 0, 0])

    logger.debug(f"The value of b1 is: {b1}")

    # calculate b2
    _, b2 = calculate_b1_b2_multilayer(n_layers, interfaces, b, materials)

    logger.debug(f"The vector b2 is: \n {b2}.")

    f0 = np.linalg.inv(M0) @ ((M2 @ np.linalg.inv(M1) @ b1) - b2)

    # F0 = np.array([[f0[0], f0[2]], [f0[2], f0[1]]])

    logger.debug(f"The vector f0 is given by: \n {f0}")

    # we need to minimize (F - F0)^T * M0 * (F - F0) where F = kappa * (1, 0, 0)
    # This leads to minimizing the following
    # function(kappa) = alpha * kappa ^ 2 - 2 * kappa * beta
    # where alpha and beta are given by:

    alpha = np.transpose(np.array([1, 0, 0])) @ M0 @ np.array([1, 0, 0])
    beta = np.transpose(np.array([1, 0, 0])) @ M0 @ f0

    logger.debug(f"beta = {beta}, alpha = {alpha}")

    # now calculating a stationary point of function, we get kappa = beta / alpha:
    kappa = beta / alpha

    logger.debug(f"kappa = {kappa}")

    calculated_radius = total_thickness / abs(kappa) / h

    logger.debug(f"calc_rad = {calculated_radius}")

    logger.debug("=== End of run ===")

    return {
        "kappa": kappa,
        "calculated_radius": calculated_radius,
        "relative_deviation": (
            (calculated_radius - measured_radius) / measured_radius
            if measured_radius
            else np.inf
        ),
        "M0": M0,
    }
