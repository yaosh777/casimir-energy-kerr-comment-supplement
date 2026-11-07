"""Compute spatial partial derivatives and the first-order correction matrix
for the Kerr metric in local coordinates using SymPy.

Run: python kerr_metric_derivatives.py
Dependency: python -m pip install sympy

Coordinate relations: phi' = phi - Omega*t, r' = r, theta' = theta,
                      dphi'/dx = 1/r0, dr'/dy = 1, dtheta'/dz = 1/r0.
Omega, r0, the mass M, and the spin parameter a are held fixed when taking
spatial partial derivatives. Differentiate first, then evaluate at r=r0 and
theta=theta0, and compute the equatorial-center special case theta0=pi/2.

The output includes partial-derivative expressions, symbolic matrices, and
LaTeX code, and is saved in the current working directory:
  kerr_metric_derivatives_output.txt
  kerr_M_matrix.tex                 (LaTeX fragment; requires amsmath)

mass denotes the mass M; M_correction denotes the first-order correction
matrix containing delta_x/y/z.
"""

from pathlib import Path
import sys

import sympy as sp


def matrix_latex(matrix, symbol_names=None):
    """Convert a matrix to LaTeX code in a pmatrix environment."""
    rows = [" & ".join(sp.latex(matrix[i, j], symbol_names=symbol_names or {})
                       for j in range(matrix.cols))
            for i in range(matrix.rows)]
    return "\\begin{pmatrix}\n" + " \\\\\n".join(rows) + "\n\\end{pmatrix}"


def algebraic_zero(expr):
    """Check whether a symbolic expression vanishes identically."""
    return sp.trigsimp(sp.cancel(expr)) == 0


def calculate():
    # Coordinate variables and the expansion center.
    r, r0 = sp.symbols("r r_0", positive=True)
    theta, theta0, phi_prime = sp.symbols("theta theta_0 phi_prime", real=True)
    mass = sp.Symbol("M", positive=True)
    a, Omega = sp.symbols("a Omega", real=True)
    delta_x, delta_y, delta_z = sp.symbols("delta_x delta_y delta_z", real=True)

    # Preserve the coordinate dependence of the auxiliary quantities.
    Sigma = r**2 + a**2 * sp.cos(theta)**2
    Delta = r**2 - 2 * mass * r + a**2
    A = (r**2 + a**2)**2 - a**2 * Delta * sp.sin(theta)**2
    omega_d = 2 * mass * a * r / A
    C_inv2 = Delta * Sigma / A * (
        1 - A**2 * sp.sin(theta)**2 / (Delta * Sigma**2)
        * (Omega - omega_d)**2
    )

    # Kerr metric in local coordinates, ordered as (t, x, y, z).
    g_original = sp.Matrix([
        [C_inv2, -A * sp.sin(theta)**2 * (Omega - omega_d) / (r0 * Sigma), 0, 0],
        [-A * sp.sin(theta)**2 * (Omega - omega_d) / (r0 * Sigma),
         -A * sp.sin(theta)**2 / (r0**2 * Sigma), 0, 0],
        [0, 0, -Sigma / Delta, 0],
        [0, 0, 0, -Sigma / r0**2],
    ])

    # Equivalent metric-component expressions; U retains its theta dependence.
    sin2 = sp.sin(theta)**2
    U = 1 - a * Omega * sin2
    g_tt = 1 - Omega**2 * (r**2 + a**2) * sin2 - 2 * mass * r * U**2 / Sigma
    g_tx = sin2 / r0 * (-Omega * (r**2 + a**2) + 2 * mass * a * r * U / Sigma)
    g_xx = -sin2 / r0**2 * (r**2 + a**2 + 2 * mass * a**2 * r * sin2 / Sigma)
    g = sp.Matrix([
        [g_tt, g_tx, 0, 0],
        [g_tx, g_xx, 0, 0],
        [0, 0, -Sigma / Delta, 0],
        [0, 0, 0, -Sigma / r0**2],
    ])
    independent = [(0, 0, "tt"), (0, 1, "tx=xt"), (1, 1, "xx"),
                   (2, 2, "yy"), (3, 3, "zz")]
    for i, j, _ in independent:
        assert algebraic_zero(g_original[i, j] - g[i, j])

    # Compute spatial partial derivatives using the chain rule.
    Jx_raw = g.diff(phi_prime) / r0
    Jy_raw = g.diff(r)
    Jz_raw = g.diff(theta) / r0
    Jx = Jx_raw.applyfunc(sp.factor_terms)
    Jy = Jy_raw.applyfunc(sp.factor_terms)
    Jz = Jz_raw.applyfunc(sp.factor_terms)

    # Evaluate at a general expansion center.
    center = {r: r0, theta: theta0}
    at_center = lambda matrix: matrix.subs(center, simultaneous=True).applyfunc(sp.factor_terms)
    Jx0, Jy0, Jz0 = (at_center(matrix) for matrix in (Jx, Jy, Jz))
    M_correction = delta_x * Jx0 + delta_y * Jy0 + delta_z * Jz0

    # Equatorial-center special case.
    Jy_equator = Jy0.subs(theta0, sp.pi / 2).applyfunc(sp.factor_terms)
    Jz_equator = Jz0.subs(theta0, sp.pi / 2).applyfunc(sp.simplify)
    M_equator = M_correction.subs(theta0, sp.pi / 2).applyfunc(sp.factor_terms)

    # Independent consistency checks: axisymmetry, matrix symmetry,
    # the t-x block determinant, and polar-derivative special cases.
    assert Jx == sp.zeros(4)
    assert M_correction == M_correction.T
    assert algebraic_zero(g_tt * g_xx - g_tx**2 + Delta * sin2 / r0**2)
    assert Jz_equator == sp.zeros(4)
    assert algebraic_zero(Jz[1, 1].subs({a: 0, Omega: 0})
                          + 2 * r**2 * sp.sin(theta) * sp.cos(theta) / r0**3)

    # Introduce abbreviations only for output, not during differentiation.
    Sigma0 = sp.Symbol("Sigma_0")
    Delta0 = sp.Symbol("Delta_0")
    U0 = sp.Symbol("U_0")
    s0, c0 = sp.symbols("s_0 c_0")
    abbrev = {
        Sigma.subs(center, simultaneous=True): Sigma0,
        Delta.subs(r, r0): Delta0,
        U.subs(theta, theta0): U0,
    }

    def for_latex(matrix):
        # Replace compound expressions before trigonometric functions
        # to preserve expression matching.
        return matrix.applyfunc(lambda e: e.xreplace(abbrev).xreplace({
            sp.sin(theta0): s0, sp.cos(theta0): c0,
        }))

    displacement_names = {delta_x: r"\delta x", delta_y: r"\delta y", delta_z: r"\delta z"}
    latex_matrix = lambda matrix: matrix_latex(for_latex(matrix), displacement_names)
    latex_parts = [
        r"% Generated by kerr_metric_derivatives.py; requires \usepackage{amsmath}.",
        r"\[s_0=\sin\theta_0,\quad c_0=\cos\theta_0,\quad"
        r"\Sigma_0=r_0^2+a^2c_0^2,\quad"
        r"\Delta_0=r_0^2-2Mr_0+a^2,\quad U_0=1-a\Omega s_0^2.\]",
        r"% General center: all derivatives have already been evaluated.",
        r"\[\left.\partial_x\hat g\right|_0=0,\qquad "
        r"R_0=\left.\partial_y\hat g\right|_0=" + latex_matrix(Jy0) + r"\]",
        r"\[P_0=\left.\partial_z\hat g\right|_0=" + latex_matrix(Jz0) + r"\]",
        r"\[\hat M=R_0\,\delta y+P_0\,\delta z="
        + latex_matrix(M_correction) + r".\]",
        r"% Additional special case: theta_0 = pi/2, not assumed before differentiation.",
        r"\[\left.\hat M\right|_{\theta_0=\pi/2}="
        + r"\delta y\," + latex_matrix(Jy_equator) + r".\]",
    ]
    latex_text = "\n\n".join(latex_parts) + "\n"

    # Standard terminal output: raw and factored differentiation results,
    # matrices, and LaTeX code.
    lines = []

    def show(title, expr=None):
        lines.append("\n" + title)
        if expr is not None:
            lines.append(sp.pretty(expr, use_unicode=True, num_columns=120))

    show("Coordinate order (t,x,y,z); Omega and r0 are fixed; r and theta are variables.")
    show("Jx=(1/r0)*∂g/∂phi', Jy=∂g/∂r, Jz=(1/r0)*∂g/∂theta.")
    for name, expr in [("Sigma", Sigma), ("Delta", Delta), ("A", A),
                       ("omega_d", omega_d), ("C^{-2}(Omega;r,theta)", C_inv2)]:
        show(name + " =", expr)
    show("Radial partial derivative of omega_d (with its coordinate dependence retained):", sp.factor_terms(sp.diff(omega_d, r)))
    show("Polar partial derivative of omega_d:", sp.factor_terms(sp.diff(omega_d, theta)))
    for i, j, name in independent:
        show("=" * 60 + "\nComponent g_" + name + ": exactly equivalent simplified form", g[i, j])
        show("x-direction derivative:", Jx[i, j])
        show("y direction: direct result of diff(g,r)", Jy_raw[i, j])
        show("y direction: factored form", Jy[i, j])
        show("z direction: direct result of diff(g,theta)/r0", Jz_raw[i, j])
        show("z direction: factored form", Jz[i, j])
    show("Radial coefficient matrix at a general center, R0 = Jy(r0,theta0):", Jy0)
    show("Polar coefficient matrix at a general center, P0 = Jz(r0,theta0):", Jz0)
    show("First-order correction matrix at a general center, M = delta_x*Jx0 + delta_y*Jy0 + delta_z*Jz0:", M_correction)
    show("Equatorial center theta0=pi/2: polar coefficient matrix", Jz_equator)
    show("First-order correction matrix at the equatorial center:", M_equator)
    show("All built-in symbolic consistency checks passed.")
    show("=" * 60 + "\nLaTeX code for the differentiated M matrix (including abbreviation definitions and the equatorial special case)\n" + latex_text)

    return "\n".join(lines) + "\n", latex_text


def main():
    # Ensure that Chinese text, Greek letters, and Unicode mathematical
    # symbols can be written on Windows.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    report, latex_text = calculate()
    output_dir = Path.cwd()
    report_path = output_dir / "kerr_metric_derivatives_output.txt"
    latex_path = output_dir / "kerr_M_matrix.tex"
    report_path.write_text(report, encoding="utf-8")
    latex_path.write_text(latex_text, encoding="utf-8")
    print(report)
    print("Standard output saved to:", report_path)
    print("Matrix LaTeX saved to:", latex_path)


if __name__ == "__main__":
    main()
