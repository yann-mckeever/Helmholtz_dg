from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from mpi4py import MPI

from .config import HelmholtzConfig
from .mesh import create_mesh
from .problem import build_problem

def solve_problem(config: HelmholtzConfig):
    """
    Solve the DG Helmholtz problem with the given configuration.
    Returns:
        uh: dolfinx.fem.Function – computed solution
        u_exact: dolfinx.fem.Function – exact solution (interpolated)
        error_L2: float – L2 error norm
    """
    # 1. Create mesh
    mesh_data = create_mesh(config.mesh)
    domain = mesh_data[0]

    # 2. Build forms and function space
    V, u, v, a, L, u_exact = build_problem(domain, config.physics)

    # 3. Create a Function to hold the solution
    uh = fem.Function(V)
    
    # 4. Solve (pass uh as the 'u' argument)
    problem = LinearProblem(a, L, u=uh, 
                            petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, 
                            petsc_options_prefix="solve_")
    uh = problem.solve()   # solves and returns the same Function

    # 5. Compute L2 error
    error = uh - u_exact
    dx = ufl.Measure("dx", domain=domain)
    M = fem.form(ufl.inner(error, error) * dx)
    error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(M), op=MPI.SUM))

    if MPI.COMM_WORLD.rank == 0:
        print(f"L2 error: {error_L2:.5e}")

    return uh, u_exact, error_L2