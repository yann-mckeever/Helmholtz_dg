from .config import HelmholtzConfig
import ufl
from dolfinx import fem
from petsc4py import PETSc
from scipy.special import hankel1
import numpy as np

def build_problem(domain, config: HelmholtzConfig):
    """Build the DG Helmholtz problem: spaces, forms, exact solution."""
    degree = config.physics.degree
    k = config.physics.k

    V = fem.functionspace(domain, ("DG", degree))
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    n = ufl.FacetNormal(domain)
    h = ufl.CellDiameter(domain)          # size of the current cell
    h_avg = (h('+') + h('-')) / 2         # average size on interior facets

    u_exact = fem.Function(V)
    

    if config.reference.exact_solution_type == "Hankel":
        def exact_solution(x):
            r = np.sqrt(x[0]**2 + x[1]**2)
            return hankel1(0, k * r)
        u_exact.interpolate(exact_solution)
        f = fem.Constant(domain, PETSc.ScalarType(0.0)) # No source inside domain
        
    elif config.reference.exact_solution_type == "StandingWave":
        # 1. Let UFL handle the continuous math perfectly at the quadrature points
        x_ufl = ufl.SpatialCoordinate(domain)
        
        # Add + 0j to strictly force the UFL graph into a complex type for PETSc
        u_sym = ufl.cos(ufl.pi * x_ufl[0]) * ufl.cos(ufl.pi * x_ufl[1]) + 0j
        
        # Exact mathematical source term
        f = (2 * ufl.pi**2 - k**2) * u_sym

        # 2. We keep the Numpy interpolation ONLY for calculating the final L2 error
        def exact_solution(x):
            val = np.cos(np.pi * x[0]) * np.cos(np.pi * x[1])
            return val.astype(PETSc.ScalarType)
            
        u_exact.interpolate(exact_solution)


    # Penalty parameters from config
    gamma_0 = config.physics.penalty_gamma_0
    i_gamma_1 = config.physics.penalty_i_gamma_1
    beta_1  = config.physics.penalty_beta_1
    sigma   = config.physics.penalty_sigma

    # ---- Interior facet terms (dS) ----
    # b_h
    a_h = (ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
           - ufl.inner(ufl.avg(ufl.dot(ufl.grad(u), n)), ufl.jump(v)) * ufl.dS
           - sigma * ufl.inner(ufl.jump(u), ufl.avg(ufl.dot(ufl.grad(v), n))) * ufl.dS)

    # J_0
    a_h += 1j * (gamma_0 / h_avg) * ufl.inner(ufl.jump(u), ufl.jump(v)) * ufl.dS

    # J_1
    du_dn = ufl.dot(ufl.grad(u), n)
    dv_dn = ufl.dot(ufl.grad(v), n)
    a_h +=  i_gamma_1 * h_avg * ufl.inner(ufl.jump(du_dn), ufl.jump(dv_dn)) * ufl.dS #No multiplication by 1j acknowledging that i_gamma_i = complex(-0.07, 0.01) 

    # L_1 (tangential jumps via projection)
    tang_u = ufl.grad(u) - du_dn * n
    tang_v = ufl.grad(v) - dv_dn * n
    a_h += 1j * (beta_1 / h_avg) * ufl.inner(ufl.jump(tang_u), ufl.jump(tang_v)) * ufl.dS

    # Helmholtz volume term
    a = a_h - k**2 * ufl.inner(u, v) * ufl.dx

    # ---- Boundary terms (ds) - using h (cell diameter), not h_avg ----
    gamma_0_bnd = gamma_0   # same penalty on boundary
    a += (- ufl.inner(du_dn, v) - ufl.inner(u, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u, v)) * ufl.ds

    L = ufl.inner(f, v) * ufl.dx
    L += (- ufl.inner(u_exact, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u_exact, v)) * ufl.ds

    return V, u, v, a, L, u_exact