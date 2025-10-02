import math
import numpy as np
from numba import njit

# -------------------- constants --------------------
SQRT1_2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)  # 1/sqrt(2π)
INV_2PI = 1.0 / (2.0 * math.pi)

# Gauss–Legendre nodes/weights on [-1,1]
_GLN = 32
_GLX, _GLW = np.polynomial.legendre.leggauss(_GLN)
_GLX = _GLX.astype(np.float64)
_GLW = _GLW.astype(np.float64)


# -------------------- JIT special functions --------------------
@njit(fastmath=True, cache=True)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x * SQRT1_2))


@njit(fastmath=True, cache=True)
def norm_pdf(x):
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(fastmath=True, cache=True)
def norm_pdf2(x, y, rho):
    t = 1.0 - rho * rho
    z = (x * x - 2.0 * rho * x * y + y * y) / (2.0 * t)
    return (INV_2PI / math.sqrt(t)) * math.exp(-z)


# -------------------- BvN CDF via Genz integral --------------------
# Φ2(h,k;ρ) = Φ(h)Φ(k) + (1/(2π)) ∫_{0}^{asin(ρ)} exp(-(h^2 - 2hk sinθ + k^2)/(2 cos^2θ)) dθ
@njit(fastmath=True, cache=True)
def _bvn_cdf(h, k, rho, GLX, GLW):
    a = math.asin(rho)  # α
    if a == 0.0:
        return norm_cdf(h) * norm_cdf(k)

    half = 0.5 * a
    s = 0.0
    # Gauss–Legendre: θ_i = half * (x_i + 1),  ∫₀^α f ≈ (α/2) Σ w_i f(θ_i)
    for i in range(GLX.size):
        xi = GLX[i]
        wi = GLW[i]
        th = half * (xi + 1.0)
        c = math.cos(th)
        st = math.sin(th)
        z = (h * h - 2.0 * h * k * st + k * k) / (2.0 * c * c)
        s += wi * math.exp(-z)
    return norm_cdf(h) * norm_cdf(k) + (INV_2PI * half) * s


@njit(fastmath=True, cache=True)
def _rect_prob(u_minus, u_plus, v_minus, v_plus, rho, GLX, GLW):
    # Rectangle mass P = Φ2(u+,v+)-Φ2(u-,v+) -Φ2(u+,v-) +Φ2(u-,v-)
    pp = _bvn_cdf(u_plus, v_plus, rho, GLX, GLW)
    mp = _bvn_cdf(u_minus, v_plus, rho, GLX, GLW)
    pm = _bvn_cdf(u_plus, v_minus, rho, GLX, GLW)
    mm = _bvn_cdf(u_minus, v_minus, rho, GLX, GLW)
    return pp - mp - pm + mm


# -------------------- Boundary assemblers (no numerical diffs) --------------------
@njit(fastmath=True, cache=True)
def _edge_L_of_u(u, v_minus, v_plus, rho):
    tau = math.sqrt(1.0 - rho * rho)
    return norm_cdf((v_plus - rho * u) / tau) - norm_cdf((v_minus - rho * u) / tau)


@njit(fastmath=True, cache=True)
def _edge_R_of_v(v, u_minus, u_plus, rho):
    tau = math.sqrt(1.0 - rho * rho)
    return norm_cdf((u_plus - rho * v) / tau) - norm_cdf((u_minus - rho * v) / tau)


@njit(fastmath=True, cache=True)
def _assemble_given_P(
    mu_i, mu_j, sigma, sigma2, rho, u_minus, u_plus, v_minus, v_plus, P
):
    # Edge terms
    L_up = _edge_L_of_u(u_plus, v_minus, v_plus, rho)
    L_um = _edge_L_of_u(u_minus, v_minus, v_plus, rho)
    R_vp = _edge_R_of_v(v_plus, u_minus, u_plus, rho)
    R_vm = _edge_R_of_v(v_minus, u_minus, u_plus, rho)

    s_u_plus, s_u_minus = norm_pdf(u_plus), norm_pdf(u_minus)
    s_v_plus, s_v_minus = norm_pdf(v_plus), norm_pdf(v_minus)
    Su = s_u_plus * L_up - s_u_minus * L_um
    Sv = s_v_plus * R_vp - s_v_minus * R_vm

    # Corner PDFs
    c_pp = norm_pdf2(u_plus, v_plus, rho)
    c_pm = norm_pdf2(u_plus, v_minus, rho)
    c_mp = norm_pdf2(u_minus, v_plus, rho)
    c_mm = norm_pdf2(u_minus, v_minus, rho)

    # Explicit “derivatives” (via PDFs)
    Du_plus, Du_minus = c_pp - c_pm, c_mp - c_mm
    Dv_plus, Dv_minus = c_pp - c_mp, c_pm - c_mm

    dSu_du = (-u_plus * s_u_plus * L_up - rho * Du_plus) + (
        u_minus * s_u_minus * L_um + rho * Du_minus
    )

    dSv_dv = (-v_plus * s_v_plus * R_vp - rho * Dv_plus) + (
        v_minus * s_v_minus * R_vm + rho * Dv_minus
    )

    dSu_dv = c_pp - c_pm - c_mp + c_mm

    # Combine
    Su_rSv = Su + rho * Sv
    Sv_rSu = Sv + rho * Su

    B00 = P
    B10 = mu_i * P - sigma * Su_rSv
    B01 = mu_j * P - sigma * Sv_rSu
    B20 = (
        (mu_i * mu_i + sigma2) * P
        - 2.0 * mu_i * sigma * Su_rSv
        + sigma2 * (dSu_du + 2.0 * rho * dSu_dv + rho * rho * dSv_dv)
    )
    B02 = (
        (mu_j * mu_j + sigma2) * P
        - 2.0 * mu_j * sigma * Sv_rSu
        + sigma2 * (dSv_dv + 2.0 * rho * dSu_dv + rho * rho * dSu_du)
    )
    # note the + rho*sigma^2*P term
    B11 = (
        (mu_i * mu_j + rho * sigma2) * P
        - mu_j * sigma * Su_rSv
        - mu_i * sigma * Sv_rSu
        + sigma2 * (rho * dSu_du + (1.0 + rho * rho) * dSu_dv + rho * dSv_dv)
    )
    return B00, B10, B01, B20, B02, B11


# -------------------- Public API --------------------
def gaussian_rect_integrals_up_to2_numba(alpha, q, b, a_ij, a_ji, L_i, L_j):
    """
    Computes I_{k,l} (k+l <= 2) on [-L_i/2,L_i/2]×[-L_j/2,L_j/2]
    using a SciPy-free, Numba-JIT implementation.
    Assumes α>0, |b|<1, finite L_i, L_j.
    Returns dict with keys: (0,0),(1,0),(0,1),(2,0),(0,2),(1,1).
    """
    rho = b
    denom = 1.0 - rho * rho
    inv_d = 1.0 / denom

    mu_i = (a_ij + rho * a_ji) * inv_d
    mu_j = (rho * a_ij + a_ji) * inv_d
    sigma2 = inv_d / (2.0 * alpha)
    sigma = math.sqrt(sigma2)

    # Standardized bounds
    half_i, half_j = 0.5 * L_i, 0.5 * L_j
    u_minus, u_plus = (-half_i - mu_i) / sigma, (half_i - mu_i) / sigma
    v_minus, v_plus = (-half_j - mu_j) / sigma, (half_j - mu_j) / sigma

    # Rectangle mass with our JIT Φ2
    P = _rect_prob(u_minus, u_plus, v_minus, v_plus, rho, _GLX, _GLW)

    # Boundary assembly (returns brackets)
    B00, B10, B01, B20, B02, B11 = _assemble_given_P(
        mu_i, mu_j, sigma, sigma2, rho, u_minus, u_plus, v_minus, v_plus, P
    )

    # Global K
    a_dot_mu = a_ij * mu_i + a_ji * mu_j
    K = (math.pi / (alpha * math.sqrt(denom))) * math.exp(-alpha * (q - a_dot_mu))

    return {
        "I00": K * B00,
        "I10": K * B10,
        "I01": K * B01,
        "I11": K * B11,
        "I20": K * B20,
        "I02": K * B02,
    }
