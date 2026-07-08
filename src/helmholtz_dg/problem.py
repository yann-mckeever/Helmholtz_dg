from .config import HelmholtzConfig
import ufl
from dolfinx import fem
from petsc4py import PETSc
from scipy.special import hankel1
import numpy as np

def build_problem(domain, config: HelmholtzConfig, facet_tags):
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
    
    elif config.reference.exact_solution_type == "Case_1":
        f = fem.Constant(domain, PETSc.ScalarType(0.0))
        ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)
        g_1 = fem.Constant(domain, PETSc.ScalarType(1.0)) # u = 1 on the Left
        g_0 = fem.Constant(domain, PETSc.ScalarType(0.0))

    elif config.reference.exact_solution_type == "PlaneWave":
        # Define the wave angle (e.g., 45 degrees)
        theta = np.pi / 4.0
        f = fem.Constant(domain, PETSc.ScalarType(1.0))
        # 1. Exact mathematical formulation for UFL boundary integrals
        x_ufl = ufl.SpatialCoordinate(domain)
        u_sym = ufl.exp(1j * k * (ufl.cos(theta) * x_ufl[0] + ufl.sin(theta) * x_ufl[1]))
        
        # 2. Numpy formulation for the L2 error interpolation
        def exact_solution(x):
            val = np.exp(1j * k * (np.cos(theta) * x[0] + np.sin(theta) * x[1]))
            return val.astype(PETSc.ScalarType)
            
        u_exact.interpolate(exact_solution)
        
        # 3. Null source term
        

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
    a_h += (gamma_0 / h_avg) * ufl.inner(ufl.jump(u), ufl.jump(v)) * ufl.dS
   
    # J_1
    du_dn = ufl.dot(ufl.grad(u), n)
    dv_dn = ufl.dot(ufl.grad(v), n)
    
    a_h +=  i_gamma_1 * h_avg * ufl.inner(ufl.jump(du_dn), ufl.jump(dv_dn)) * ufl.dS #No multiplication by 1j acknowledging that i_gamma_i = complex(-0.07, 0.01) 
   
    # L_1 (tangential jumps via projection)
    tang_u = ufl.grad(u) - du_dn * n
    tang_v = ufl.grad(v) - dv_dn * n
    
    a_h += (beta_1 / h_avg) * ufl.inner(ufl.jump(tang_u), ufl.jump(tang_v)) * ufl.dS 


    # Helmholtz volume term
    a = a_h - k**2 * ufl.inner(u, v) * ufl.dx  

    # ---- Boundary terms (ds) - using h (cell diameter), not h_avg ----
    gamma_0_bnd = gamma_0   # same penalty on boundary

    L = ufl.inner(f, v) * ufl.dx

    if config.reference.exact_solution_type == "Case_1":
        a += (- ufl.inner(du_dn, v) - ufl.inner(u, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u, v)) * ds(1)
        a += (- ufl.inner(du_dn, v) - ufl.inner(u, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u, v)) * ds(2)

        L += (- ufl.inner(g_1, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(g_1, v)) * ds(1)
        L += (- ufl.inner(g_0, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(g_0, v)) * ds(2)
    
    elif config.reference.exact_solution_type == "PlaneWave":
        # Apply the exact plane wave as the boundary condition on all edges
        a += (- ufl.inner(du_dn, v) - ufl.inner(u, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u, v)) * ufl.ds
        L += (- ufl.inner(u_exact, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u_exact, v)) * ufl.ds

        #Robin

        # a += (- ufl.inner(du_dn, v) - ufl.inner(u, dv_dn) + (1j*k + gamma_0_bnd/h) * ufl.inner(u, v)) * ufl.ds
        # L += (- ufl.inner(u_exact, dv_dn) + (1j*k + gamma_0_bnd/h) * ufl.inner(u_exact, v)) * ufl.ds

    else :
        a += (- ufl.inner(du_dn, v) - ufl.inner(u, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u, v)) * ufl.ds
        L += (- ufl.inner(u_exact, dv_dn) + 1j * (gamma_0_bnd / h) * ufl.inner(u_exact, v)) * ufl.ds

        #Robin
        # a += (- ufl.inner(du_dn, v) - ufl.inner(u, dv_dn) + (1j*k + gamma_0_bnd/h) * ufl.inner(u, v)) * ufl.ds
        # L += (- ufl.inner(u_exact, dv_dn) + (1j*k + gamma_0_bnd/h) * ufl.inner(u_exact, v)) * ufl.ds

    return V, u, v, a, L, u_exact
