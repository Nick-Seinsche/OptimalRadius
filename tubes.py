# Standard
import logging

# Third Party
import numpy as np

logger = logging.getLogger(__name__)


def calculate_Q_2_isotropic(youngs_modulus: float, poisson_ratio: float) -> np.ndarray:
    """
    Calculates the matrix representation of the bilinearform Q_2
    for the top and bottom layer under the assumption of isotropy.
    Since we only have elasticity measurements for the
    whole leaf we have M_top == M_bot.

    Args:
        youngs_modulus: float
        poisson_ratio: float

    Returns:
        A 3x3 matrix representing the bilinearform of Q_2.

    """
    logger.debug(f"youngs_modulus={youngs_modulus}, poisson_ratio={poisson_ratio}")

    # calculate the lame parameters:
    lambd = (
        youngs_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
    )
    mu = youngs_modulus / (2 * (1 + poisson_ratio))

    logger.debug(f"lambda = {lambd}, mu={mu}")

    # calculate Q2, which is the linearized strain at the identity.
    def Q2(G):
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
    M[0, 0] = Q2(e1)
    M[1, 1] = Q2(e2)
    M[2, 2] = Q2(e3)
    M[0, 1] = 0.5 * (Q2(e1 + e2) - Q2(e1) - Q2(e2))
    M[1, 0] = M[0, 1]
    M[0, 2] = 0.5 * (Q2(e1 + e3) - Q2(e1) - Q2(e3))
    M[2, 0] = M[0, 2]
    M[1, 2] = 0.5 * (Q2(e2 + e3) - Q2(e2) - Q2(e3))
    M[2, 1] = M[1, 2]

    logger.debug(f"The matrix M is given by {M}")

    return (M, M)


def calculate_Q_2_bar(
    tau: float,
    M_top: np.ndarray,
    M_bot: np.ndarray,
) -> np.ndarray:
    """Calculates the matrix representation of the quadratic form Q_2_bar"""
    M1 = (tau + 1 / 2) * M_bot + (1 / 2 - tau) * M_top
    M2 = (0.5 * tau**2 - 0.5 * 1 / 2**2) * M_bot + (
        0.5 * 1 / 2**2 - 0.5 * tau**2
    ) * M_top
    M3 = (1 / 3 * tau**3 + 1 / 3 * 1 / 2**3) * M_bot + (
        1 / 3 * 1 / 2**3 - 1 / 3 * tau**3
    ) * M_top
    M0 = M3 - (M2 @ np.linalg.inv(M1)) @ M2
    return M0, M1, M2, M3


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

    logger.debug("=== Running optimal radius ===")
    logger.debug(f"Input parameters: tau={tau}, h={h}, d={total_thickness}, "
                 f"btop={btop}, bbot={bbot}")

    if isinstance(youngs_modulus, list) and isinstance(poisson_ratio, list):
        M_bot, _ = calculate_Q_2_isotropic(youngs_modulus[0], poisson_ratio[0])
        M_top, _ = calculate_Q_2_isotropic(youngs_modulus[1], poisson_ratio[1])
    else:
        M_bot, M_top = calculate_Q_2_isotropic(youngs_modulus, poisson_ratio)

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
