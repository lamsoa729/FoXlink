import numpy as np
from dataclasses import dataclass
from typing import Dict
from scipy.stats import norm, multivariate_normal

# ---------- helpers: ϕ, Φ, ϕ2, Φ2 and their needed partials ----------


def phi(z):
    # standard normal pdf
    return norm.pdf(z)


def Phi(z):
    # standard normal cdf
    return norm.cdf(z)


def phi2(x, y, rho):
    # bivariate standard normal pdf with correlation rho
    denom = 2 * np.pi * np.sqrt(1 - rho**2)
    q = (x**2 - 2 * rho * x * y + y**2) / (2 * (1 - rho**2))
    return np.exp(-q) / denom


def Phi2(x, y, rho):
    # bivariate standard normal cdf with correlation rho
    mean = np.array([0.0, 0.0])
    cov = np.array([[1.0, rho], [rho, 1.0]])
    # multivariate_normal.cdf is a special function (no manual quadrature here)
    return multivariate_normal(mean=mean, cov=cov).cdf(np.array([x, y]))


def dPhi2_dx(x, y, rho):
    # ∂Φ2/∂x = ϕ(x) * Φ( (y - ρ x)/sqrt(1-ρ²) )
    t = (y - rho * x) / np.sqrt(1 - rho**2)
    return phi(x) * Phi(t)


def dPhi2_dy(x, y, rho):
    # ∂Φ2/∂y = ϕ(y) * Φ( (x - ρ y)/sqrt(1-ρ²) )
    t = (x - rho * y) / np.sqrt(1 - rho**2)
    return phi(y) * Phi(t)


def d2Phi2_dxx(x, y, rho):
    # ∂²Φ2/∂x² = -x ϕ(x) Φ((y-ρx)/√(1-ρ²)) - (ρ/√(1-ρ²)) ϕ2(x,y;ρ)
    return -x * phi(x) * Phi((y - rho * x) / np.sqrt(1 - rho**2)) - (
        rho / np.sqrt(1 - rho**2)
    ) * phi2(x, y, rho)


def d2Phi2_dyy(x, y, rho):
    # ∂²Φ2/∂y² = -y ϕ(y) Φ((x-ρy)/√(1-ρ²)) - (ρ/√(1-ρ²)) ϕ2(x,y;ρ)
    return -y * phi(y) * Phi((x - rho * y) / np.sqrt(1 - rho**2)) - (
        rho / np.sqrt(1 - rho**2)
    ) * phi2(x, y, rho)


def d2Phi2_dxy(x, y, rho):
    # ∂²Φ2/∂x∂y = ϕ2(x,y;ρ)
    return phi2(x, y, rho)


# ---------- main evaluator ----------


@dataclass
class Params:
    a: float
    r: float
    b: float
    c: float
    d: float
    L_i: float
    L_j: float


def _four_corners(zix, ziy, zjx, zjy):
    # returns (x+, x-, y+, y-) to avoid confusion
    return (zix, ziy, zjx, zjy)


def _inclusion_exclusion(func, zi_plus, zi_minus, zj_plus, zj_minus):
    # S[f] = f(zi+,zj+) - f(zi-,zj+) - f(zi+,zj-) + f(zi-,zj-)
    return (
        func(zi_plus, zj_plus)
        - func(zi_minus, zj_plus)
        - func(zi_plus, zj_minus)
        + func(zi_minus, zj_minus)
    )


def compute_Ik_l_all(p: Params) -> Dict[str, float]:
    """
    Returns a dict with keys: I00, I10, I01, I20, I02, I11
    corresponding to I_{0,0}, I_{1,0}, I_{0,1}, I_{2,0}, I_{0,2}, I_{1,1}.
    """
    a, r, b, c, d, L_i, L_j = p.a, p.r, p.b, p.c, p.d, p.L_i, p.L_j
    if not (abs(d) < 1):
        raise ValueError("Require |d|<1 for positive-definite quadratic form.")
    if a <= 0 or r <= 0 or L_i <= 0 or L_j <= 0:
        raise ValueError("Require a>0, r>0, L_i>0, L_j>0.")

    rho = d
    one_minus_d2 = 1 - d**2
    sigma = np.sqrt(1.0 / (2 * a * one_minus_d2))

    # mean shift from completing the square
    mu_i = -r * (b + d * c) / one_minus_d2
    mu_j = -r * (d * b + c) / one_minus_d2

    # standardized bounds
    zi_plus = (+L_i / 2 - mu_i) / sigma
    zi_minus = (-L_i / 2 - mu_i) / sigma
    zj_plus = (+L_j / 2 - mu_j) / sigma
    zj_minus = (-L_j / 2 - mu_j) / sigma

    # rectangle mass Δ
    Delta = _inclusion_exclusion(
        lambda x, y: Phi2(x, y, rho), zi_plus, zi_minus, zj_plus, zj_minus
    )

    # global prefactor P
    expo = -a * r**2 + (a * r**2 / one_minus_d2) * (b**2 + 2 * d * b * c + c**2)
    P = (np.pi / (a * np.sqrt(one_minus_d2))) * np.exp(expo)

    # I00
    I00 = P * Delta

    # derivatives of log-prefactor E = log(P a sqrt(1-d^2)/pi) = expo
    Eb = (2 * a * r**2 / one_minus_d2) * (b + d * c)
    Ec = (2 * a * r**2 / one_minus_d2) * (c + d * b)
    Ebb = 2 * a * r**2 / one_minus_d2
    Ecc = 2 * a * r**2 / one_minus_d2
    Ebc = (2 * a * r**2 / one_minus_d2) * d

    # how z-bounds move with b,c (same for + and -)
    alpha_ib = r * np.sqrt(2 * a / one_minus_d2)  # ∂z_i/∂b
    alpha_jb = r * d * np.sqrt(2 * a / one_minus_d2)  # ∂z_j/∂b
    alpha_ic = r * d * np.sqrt(2 * a / one_minus_d2)  # ∂z_i/∂c
    alpha_jc = r * np.sqrt(2 * a / one_minus_d2)  # ∂z_j/∂c

    # first-derivative corner sums
    S_b = _inclusion_exclusion(
        lambda x, y: alpha_ib * dPhi2_dx(x, y, rho) + alpha_jb * dPhi2_dy(x, y, rho),
        zi_plus,
        zi_minus,
        zj_plus,
        zj_minus,
    )
    S_c = _inclusion_exclusion(
        lambda x, y: alpha_ic * dPhi2_dx(x, y, rho) + alpha_jc * dPhi2_dy(x, y, rho),
        zi_plus,
        zi_minus,
        zj_plus,
        zj_minus,
    )

    # I10 = -(2ar)^(-1) * ( Eb*I00 + P * S_b )
    I10 = -(Eb * I00 + P * S_b) / (2 * a * r)

    # I01 = -(2ar)^(-1) * ( Ec*I00 + P * S_c )
    I01 = -(Ec * I00 + P * S_c) / (2 * a * r)

    # second-derivative corner sums
    S_bb = _inclusion_exclusion(
        lambda x, y: (alpha_ib**2) * d2Phi2_dxx(x, y, rho)
        + 2 * alpha_ib * alpha_jb * d2Phi2_dxy(x, y, rho)
        + (alpha_jb**2) * d2Phi2_dyy(x, y, rho),
        zi_plus,
        zi_minus,
        zj_plus,
        zj_minus,
    )
    S_cc = _inclusion_exclusion(
        lambda x, y: (alpha_ic**2) * d2Phi2_dxx(x, y, rho)
        + 2 * alpha_ic * alpha_jc * d2Phi2_dxy(x, y, rho)
        + (alpha_jc**2) * d2Phi2_dyy(x, y, rho),
        zi_plus,
        zi_minus,
        zj_plus,
        zj_minus,
    )
    S_bc = _inclusion_exclusion(
        lambda x, y: (alpha_ib * alpha_ic) * d2Phi2_dxx(x, y, rho)
        + (alpha_ib * alpha_jc + alpha_jb * alpha_ic) * d2Phi2_dxy(x, y, rho)
        + (alpha_jb * alpha_jc) * d2Phi2_dyy(x, y, rho),
        zi_plus,
        zi_minus,
        zj_plus,
        zj_minus,
    )

    # I20 = (2ar)^(-2) * [ (Eb^2 + Ebb) I00 + 2 Eb P S_b + P S_bb ]
    I20 = ((Eb**2 + Ebb) * I00 + 2 * Eb * P * S_b + P * S_bb) / (2 * a * r) ** 2

    # I02 = (2ar)^(-2) * [ (Ec^2 + Ecc) I00 + 2 Ec P S_c + P S_cc ]
    I02 = ((Ec**2 + Ecc) * I00 + 2 * Ec * P * S_c + P * S_cc) / (2 * a * r) ** 2

    # I11 = (2ar)^(-2) * [ (Eb Ec + Ebc) I00 + Eb P S_c + Ec P S_b + P S_bc ]
    I11 = ((Eb * Ec + Ebc) * I00 + Eb * P * S_c + Ec * P * S_b + P * S_bc) / (
        2 * a * r
    ) ** 2

    return {
        "I00": float(I00),
        "I10": float(I10),
        "I01": float(I01),
        "I11": float(I11),
        "I20": float(I20),
        "I02": float(I02),
    }


# ---------- example usage ----------
if __name__ == "__main__":
    # sample parameters (feel free to edit)
    p = Params(a=1.7, r=0.9, b=0.2, c=-0.4, d=0.3, L_i=3.0, L_j=2.5)
    vals = compute_Ik_l_all(p)
    for k in [
        "I00",
        "I10",
        "I01",
        "I11",
        "I20",
        "I02",
    ]:
        print(k, vals[k])
