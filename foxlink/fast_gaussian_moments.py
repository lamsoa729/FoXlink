import numpy as np
from math import sqrt, pi
from math import comb as _comb  # Python 3.8+
from scipy.special import erf  # vectorized, also fine for scalars


def gaussian_moment_symbolic(Li, Lj, a, r, k, l, b=1):
    """
    I_{k,l} = ∬_{[-Li/2,Li/2]×[-Lj/2,Lj/2]} s_i^k s_j^l exp(-a (r + s_i - b s_j)^2) ds_i ds_j
    Fully closed form for k,l ∈ {0,1,2}. No numerical differentiation or quadrature.

    Uses the decomposition u = r + s_i - s_j with u ∈ [r-Li/2, r+Li/2] and inner moments over s_j.
    The sign b ∈ {±1} contributes a simple factor b^l due to symmetric s_j limits.
    """
    if k not in (0, 1, 2) or l not in (0, 1, 2):
        raise ValueError("This function supports k,l ∈ {0,1,2}.")
    if b == 0:
        raise ValueError("b must be +1 or -1")
    b = 1 if b > 0 else -1

    # a=0 special case: exponential=1 and integral factorizes exactly
    if a == 0.0:

        def seg_moment(L, n):
            if n % 2 == 1:
                return 0.0
            return (L / 2.0) ** (n + 1) * 2.0 / (n + 1)

        return (b**l) * seg_moment(Li, k) * seg_moment(Lj, l)

    c = sqrt(a)
    v = Lj / 2.0
    u1 = r - Li / 2.0
    u2 = r + Li / 2.0

    # Basic building blocks
    def E_sum(u):  # erf(c(v-u)) + erf(c(v+u))
        return erf(c * (v - u)) + erf(c * (v + u))

    def E_diff(u):  # erf(c(v-u)) - erf(c(v+u))
        return erf(c * (v - u)) - erf(c * (v + u))

    def G1(u):  # exp(-a (v-u)^2)
        x = v - u
        return float(np.exp(-a * x * x))

    def G2(u):  # exp(-a (v+u)^2)
        x = v + u
        return float(np.exp(-a * x * x))

    # Primitives ∫ x^p e^{-a x^2} dx evaluated at x
    def P0(x):  # p=0
        return (sqrt(pi) / (2.0 * c)) * erf(c * x)

    def P1(x):  # p=1
        return -np.exp(-a * x * x) / (2.0 * a)

    def P2(x):  # p=2
        return (sqrt(pi) / (4.0 * a**1.5)) * erf(c * x) - x * np.exp(-a * x * x) / (
            2.0 * a
        )

    def P3(x):  # p=3
        G = np.exp(-a * x * x)
        return -G * (x * x / (2.0 * a) + 1.0 / (2.0 * a * a))

    # Binomial helpers for integrals of u^m times shifted Gaussians
    def J_m(u, m):
        """J_m(u) = ∫ u^m [G(v-u) - G(v+u)] du (indefinite primitive, up to constant)."""
        # First term: ∫ u^m G(v-u) du = - Σ_{p=0..m} C(m,p) v^{m-p} (-1)^p P_p(v-u)
        s1 = 0.0
        x1 = v - u
        for p in range(m + 1):
            s1 += _comb(m, p) * (v ** (m - p)) * ((-1.0) ** p) * P_eval(p, x1)
        s1 = -s1
        # Second term: ∫ u^m G(v+u) du = Σ_{p=0..m} C(m,p) (-v)^{m-p} P_p(v+u)
        s2 = 0.0
        x2 = v + u
        for p in range(m + 1):
            s2 += _comb(m, p) * ((-v) ** (m - p)) * P_eval(p, x2)
        return s1 - s2

    def K_m(u, m):
        """K_m(u) = ∫ u^m [G(v-u) + G(v+u)] du (indefinite primitive)."""
        s1 = 0.0
        x1 = v - u
        for p in range(m + 1):
            s1 += _comb(m, p) * (v ** (m - p)) * ((-1.0) ** p) * P_eval(p, x1)
        s1 = -s1
        s2 = 0.0
        x2 = v + u
        for p in range(m + 1):
            s2 += _comb(m, p) * ((-v) ** (m - p)) * P_eval(p, x2)
        return s1 + s2

    def R_m(u, m):
        """R_m(u) = ∫ u^m [(v-u)G(v-u) + (v+u)G(v+u)] du (indefinite primitive)."""
        # Similar binomial, but with one higher power in P (p+1)
        s1 = 0.0
        x1 = v - u
        for p in range(m + 1):
            s1 += _comb(m, p) * (v ** (m - p)) * ((-1.0) ** p) * P_eval(p + 1, x1)
        s1 = -s1
        s2 = 0.0
        x2 = v + u
        for p in range(m + 1):
            s2 += _comb(m, p) * ((-v) ** (m - p)) * P_eval(p + 1, x2)
        return s1 + s2

    def P_eval(p, x):
        if p == 0:
            return P0(x)
        if p == 1:
            return P1(x)
        if p == 2:
            return P2(x)
        if p == 3:
            return P3(x)
        raise ValueError("Internal error: P_p only implemented for p ≤ 3")

    # Recurrences for ∫ u^n * [E_sum or E_diff] du
    def F_sum(n, u):
        """F_sum(n,u) = ∫ u^n [E(v-u)+E(v+u)] du."""
        D = G1(u) - G2(u)
        V = u * E_sum(u) + D / (c * sqrt(pi))
        if n == 0:
            return V
        return (u**n * V - (n / (c * sqrt(pi))) * J_m(u, n - 1)) / (n + 1)

    def F_diff(n, u):
        """F_diff(n,u) = ∫ u^n [E(v-u)-E(v+u)] du."""
        S = G1(u) + G2(u)
        V = u * E_diff(u) - S / (c * sqrt(pi))
        if n == 0:
            return V
        return (u**n * V - (n / (c * sqrt(pi))) * K_m(u, n - 1)) / (n + 1)

    # A_n(u) = ∫ u^n S0(u) du, where S0(u) = (√π/(2c)) * E_sum(u)
    def A(n, u):
        return (sqrt(pi) / (2.0 * c)) * F_sum(n, u)

    # B_n(u) = ∫ u^n S2(u) du, with
    # S2(u) = u^2 S0(u) - (u/a)[G1-G2] + (√π/(4 a^{3/2})) [E_diff] - (1/(2a))[(v-u)G1+(v+u)G2]
    def B(n, u):
        return (
            A(n + 2, u)
            - (1.0 / a) * J_m(u, n + 1)
            + (sqrt(pi) / (4.0 * a**1.5)) * F_diff(n, u)
            - (1.0 / (2.0 * a)) * R_m(u, n)
        )

    # Boundary difference Δ[f] = f(u2) - f(u1)
    def Delta(func):
        return func(u2) - func(u1)

    # Assemble the nine cases (k,l ∈ {0,1,2})
    if l == 0:
        if k == 0:
            base = Delta(lambda u: A(0, u))
        elif k == 1:
            base = Delta(lambda u: A(1, u) - r * A(0, u))
        else:  # k == 2
            base = Delta(lambda u: A(2, u) - 2.0 * r * A(1, u) + r * r * A(0, u))
    elif l == 1:
        if k == 0:
            base = Delta(lambda u: A(1, u) - (1.0 / (2.0 * a)) * J_m(u, 0))
        elif k == 1:
            base = Delta(
                lambda u: A(2, u) - r * A(1, u) - (1.0 / (2.0 * a)) * J_m(u, 1)
            )
        else:  # k == 2
            base = Delta(
                lambda u: A(3, u)
                - 2.0 * r * A(2, u)
                + r * r * A(1, u)
                - (1.0 / (2.0 * a)) * J_m(u, 2)
            )
    else:  # l == 2
        if k == 0:
            base = Delta(lambda u: B(0, u))
        elif k == 1:
            base = Delta(lambda u: B(1, u) - r * B(0, u))
        else:  # k == 2
            base = Delta(lambda u: B(2, u) - 2.0 * r * B(1, u) + r * r * B(0, u))

    return (b**l) * float(base)


# --- quick sanity checks (optional) ---
if __name__ == "__main__":
    # symmetry: r=0, odd powers vanish when b=+1
    print("I_10 (r=0) ~ 0:", gaussian_moment_symbolic(3.0, 2.0, 1.2, 0.0, 1, 0, b=+1))
    print("I_01 (r=0) ~ 0:", gaussian_moment_symbolic(3.0, 2.0, 1.2, 0.0, 0, 1, b=+1))
    # b flip when l is odd
    v1 = gaussian_moment_symbolic(3.0, 3.0, 0.7, 0.5, 0, 1, b=+1)
    v2 = gaussian_moment_symbolic(3.0, 3.0, 0.7, 0.5, 0, 1, b=-1)
    print("b flip ratio (should be -1):", v2 / v1)
    # a→0 factorization
    exact = ((2.0 / 2.0) ** 3 * 2.0 / 3.0) * (
        (4.0 / 2.0) ** 1 * 2.0 / 1.0
    )  # I_20 with a=0
    print(
        "a=0 check I_20:",
        gaussian_moment_symbolic(2.0, 4.0, 0.0, 0.3, 2, 0, b=+1),
        "vs",
        exact,
    )
