import math
import numpy as np
from numba import njit

# -------------------- constants --------------------
SQRT1_2 = 1.0 / math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)  # 1/sqrt(2π)
INV_2PI = 1.0 / (2.0 * math.pi)
TWO_PI = 2.0 * math.pi

# Half-range Gauss–Legendre nodes/weights used by Drezner–Wesolowsky (via Genz’ variant).
# Columns correspond to |ρ|<0.3  -> 3 nodes,
#                       |ρ|<0.75 -> 6 nodes,
#                       else     -> 10 nodes.
_DW_X = np.array(
    [
        [-0.9324695142031522, -0.9815606342467191, -0.9931285991850949],
        [-0.6612093864662647, -0.9041172563704750, -0.9639719272779138],
        [-0.2386191860831970, -0.7699026741943050, -0.9122344282513259],
        [0.0000000000000000, -0.5873179542866171, -0.8391169718222188],
        [0.0000000000000000, -0.3678314989981802, -0.7463319064601508],
        [0.0000000000000000, -0.1252334085114692, -0.6360536807265150],
        [0.0000000000000000, 0.0000000000000000, -0.5108670019508271],
        [0.0000000000000000, 0.0000000000000000, -0.3737060887154196],
        [0.0000000000000000, 0.0000000000000000, -0.2277858511416451],
        [0.0000000000000000, 0.0000000000000000, -0.07652652113349733],
    ],
    dtype=np.float64,
)

_DW_W = np.array(
    [
        [0.1713244923791705, 0.04717533638651177, 0.01761400713915212],
        [0.3607615730481384, 0.1069393259953183, 0.04060142980038694],
        [0.4679139345726904, 0.1600783285433464, 0.06267204833410906],
        [0.0000000000000000, 0.2031674267230659, 0.08327674157670475],
        [0.0000000000000000, 0.2334925365383547, 0.1019301198172404],
        [0.0000000000000000, 0.2491470458134029, 0.1181945319615184],
        [0.0000000000000000, 0.0000000000000000, 0.1316886384491766],
        [0.0000000000000000, 0.0000000000000000, 0.1420961093183821],
        [0.0000000000000000, 0.0000000000000000, 0.1491729864726037],
        [0.0000000000000000, 0.0000000000000000, 0.1527533871307259],
    ],
    dtype=np.float64,
)


# -------------------- JIT special functions --------------------
@njit(fastmath=True, cache=True)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x * SQRT1_2))


@njit(fastmath=True, cache=True)
def norm_pdf(x):
    return INV_SQRT_2PI * math.exp(-0.5 * x * x)


@njit(fastmath=True, cache=True)
def phi2(x, y, rho):
    t = 1.0 - rho * rho
    z = (x * x - 2.0 * rho * x * y + y * y) / (2.0 * t)
    return (INV_2PI / math.sqrt(t)) * math.exp(-z)


# -------------------- Φ₂ via Drezner–Wesolowsky (Genz’ variant) --------------------
@njit(fastmath=True, cache=True)
def _bvn_cdf_dw(h, k, r, X, W):
    """DW/Genz bivariate normal CDF Φ₂(h,k; r)."""
    ab = abs(r)
    # select table and length
    if ab < 0.3:
        ng = 0
        lg = 3
    elif ab < 0.75:
        ng = 1
        lg = 6
    else:
        ng = 2
        lg = 10

    h2 = h * h
    k2 = k * k
    hk = h * k
    bvn = 0.0

    if ab < 0.925:
        hs = 0.5 * (h2 + k2)
        asr = math.asin(r)
        half = 0.5
        # sum over symmetric nodes: is = -1, +1
        for i in range(lg):
            xi = X[i, ng]
            wi = W[i, ng]
            for sgn in (-1.0, 1.0):
                sn = math.sin(asr * (sgn * xi + 1.0) * half)
                den = 1.0 - sn * sn
                bvn += wi * math.exp((sn * hk - hs) / den)
        bvn = bvn * asr / TWO_PI + norm_cdf(-h) * norm_cdf(-k)
        return bvn

    # high-correlation branch
    if r < 0.0:
        k = -k
        hk = -hk

    if ab < 1.0:
        as_ = (1.0 - r) * (1.0 + r)
        a = math.sqrt(as_)
        bs = (h - k) * (h - k)
        c = (4.0 - hk) * 0.125
        d = (12.0 - hk) * 0.0625  # = (12 - hk)/16
        asr = -(bs / as_ + hk) * 0.5

        if asr > -100.0:
            bvn = (
                a
                * math.exp(asr)
                * (
                    1.0
                    - c * (bs - as_) * (1.0 - d * bs / 5.0) / 3.0
                    + c * d * as_ * as_ / 5.0
                )
            )

        if -hk < 100.0:
            b = math.sqrt(bs)
            bvn -= (
                math.exp(-0.5 * hk)
                * math.sqrt(TWO_PI)
                * norm_cdf(-b / a)
                * b
                * (1.0 - c * bs * (1.0 - d * bs / 5.0) / 3.0)
            )

        a *= 0.5
        for i in range(lg):
            xi = X[i, ng]
            wi = W[i, ng]
            for sgn in (-1.0, 1.0):
                xs = (a * (sgn * xi + 1.0)) ** 2
                rs = math.sqrt(1.0 - xs)
                asr = -(bs / xs + hk) * 0.5
                if asr > -100.0:
                    term = math.exp(asr) * (
                        math.exp(-hk * (1.0 - rs) / (2.0 * (1.0 + rs))) / rs
                        - (1.0 + c * xs * (1.0 + d * xs))
                    )
                    bvn += a * wi * term

        bvn = -bvn / TWO_PI

    if r > 0.0:
        bvn += norm_cdf(-max(h, k))
    if r < 0.0:
        tmp = norm_cdf(-h) - norm_cdf(-k)
        if tmp > 0.0:
            bvn = -bvn + tmp
        else:
            bvn = -bvn

    return bvn


@njit(fastmath=True, cache=True)
def _rect_prob_dw(u_minus, u_plus, v_minus, v_plus, rho, X, W):
    # P{U ≤ u+, V ≤ v+} - P{U ≤ u-, V ≤ v+} - P{U ≤ u+, V ≤ v-} + P{U ≤ u-, V ≤ v-}
    pp = _bvn_cdf_dw(u_plus, v_plus, rho, X, W)
    mp = _bvn_cdf_dw(u_minus, v_plus, rho, X, W)
    pm = _bvn_cdf_dw(u_plus, v_minus, rho, X, W)
    mm = _bvn_cdf_dw(u_minus, v_minus, rho, X, W)
    return pp - mp - pm + mm


# -------------------- Boundary assemblers (unchanged) --------------------
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
    c_pp = phi2(u_plus, v_plus, rho)
    c_pm = phi2(u_plus, v_minus, rho)
    c_mp = phi2(u_minus, v_plus, rho)
    c_mm = phi2(u_minus, v_minus, rho)

    # Explicit boundary combos (no numeric diffs)
    Du_plus, Du_minus = c_pp - c_pm, c_mp - c_mm
    Dv_plus, Dv_minus = c_pp - c_mp, c_pm - c_mm

    dSu_du = (-u_plus * s_u_plus * L_up - rho * Du_plus) + (
        u_minus * s_u_minus * L_um + rho * Du_minus
    )

    dSv_dv = (-v_plus * s_v_plus * R_vp - rho * Dv_plus) + (
        v_minus * s_v_minus * R_vm + rho * Dv_minus
    )

    dSu_dv = c_pp - c_pm - c_mp + c_mm

    # Combine (without global K)
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
    # NOTE: + rho * sigma^2 * P term is essential
    B11 = (
        (mu_i * mu_j + rho * sigma2) * P
        - mu_j * sigma * Su_rSv
        - mu_i * sigma * Sv_rSu
        + sigma2 * (rho * dSu_du + (1.0 + rho * rho) * dSu_dv + rho * dSv_dv)
    )
    return B00, B10, B01, B20, B02, B11


# -------------------- Public API (DW Φ₂) --------------------
def gaussian_rect_integrals_up_to2_numba_dw(alpha, q, b, a_ij, a_ji, L_i, L_j):
    """
    Uses Drezner–Wesolowsky (1990) Φ₂ with tiny fixed GL tables (3/6/10 nodes).
    Assumes α>0, |b|<1, finite L_i, L_j.
    Returns dict for (k,l) with k+l <= 2.
    """
    rho = b
    denom = 1.0 - rho * rho
    inv_d = 1.0 / denom

    mu_i = (a_ij + rho * a_ji) * inv_d
    mu_j = (rho * a_ij + a_ji) * inv_d
    sigma2 = inv_d / (2.0 * alpha)
    sigma = math.sqrt(sigma2)

    # standardize bounds
    half_i, half_j = 0.5 * L_i, 0.5 * L_j
    u_minus, u_plus = (-half_i - mu_i) / sigma, (half_i - mu_i) / sigma
    v_minus, v_plus = (-half_j - mu_j) / sigma, (half_j - mu_j) / sigma

    # rectangle mass via DW Φ₂
    P = _rect_prob_dw(u_minus, u_plus, v_minus, v_plus, rho, _DW_X, _DW_W)

    # assemble brackets
    B00, B10, B01, B20, B02, B11 = _assemble_given_P(
        mu_i, mu_j, sigma, sigma2, rho, u_minus, u_plus, v_minus, v_plus, P
    )

    # global K
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
