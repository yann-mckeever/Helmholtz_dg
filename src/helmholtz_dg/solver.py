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
    V, u, v, a, L, u_exact = build_problem(domain, config)

    # 3. Create a Function to hold the solution
    uh = fem.Function(V)
    
    #4 Configure PETSc Options based on config
    petsc_opts = {}
    if config.solver.solver_type=="direct":
        petsc_opts["ksp_type"] = "preonly"
        petsc_opts["pc_type"] = "lu"
        if config.solver.use_mumps:
            petsc_opts["pc_factor_mat_solver_type"] = "mumps"
    
    elif config.solver.solver_type == "gmres":
        petsc_opts["ksp_type"] = "gmres"
        petsc_opts["pc_type"] = config.solver.preconditioner  # e.g., "ilu" or "jacobi"
        petsc_opts["ksp_rtol"] = 1e-6                         # Relative tolerance
        petsc_opts["ksp_max_it"] = 1000                       # Max iterations
        petsc_opts["ksp_monitor"] = ""                        # Print residual at each step
    else:
        raise ValueError(f"Unknown solver_type: {config.solver.solver_type}")

    if MPI.COMM_WORLD.rank == 0:
        print(f"\n--- Solving with {config.solver.solver_type.upper()} "
              f"(PC: {config.solver.preconditioner.upper()}) ---")


    #5 Solve the linear system
    problem = LinearProblem(a, L, u=uh, 
                            petsc_options=petsc_opts, 
                            petsc_options_prefix="solve_")
    uh = problem.solve()   # solves and returns the same Function

    # 6. Compute L2 error
    error = uh - u_exact
    dx = ufl.Measure("dx", domain=domain)
    M = fem.form(ufl.inner(error, error) * dx)
    error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(M), op=MPI.SUM))

    if MPI.COMM_WORLD.rank == 0:
        print(f"L2 error: {error_L2:.5e}")

    return uh, u_exact, error_L2