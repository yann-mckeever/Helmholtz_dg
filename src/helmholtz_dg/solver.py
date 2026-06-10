from dolfinx import fem
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np
from mpi4py import MPI

from .config import HelmholtzConfig
from .mesh import create_mesh
from .problem import build_problem

from petsc4py import PETSc
from .ddm import build_subdomains
from .preconditioner import OptimizedSchwarzPC

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

    # 4. Assemble the Global Matrix manually (required for the PC)
    A = fem.petsc.assemble_matrix(fem.form(a))
    A.assemble()
    b = fem.petsc.assemble_vector(fem.form(L))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    b.assemble()
    
    # 5. Configure PETSc Options
    ksp = PETSc.KSP().create(domain.comm)
    ksp.setOperators(A)

    if config.solver.solver_type == "direct":
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")

    elif config.solver.solver_type == "gmres":
        ksp.setType("fgmres")
        ksp.setTolerances(rtol=1e-6, atol=1e-10, max_it=1000)  
        ksp.setMonitor(lambda ksp, its, rnorm: print(f"Iteration {its}: Residual = {rnorm:.4e}"))

        pc = ksp.getPC()

        if config.solver.preconditioner == "custom_asm":
            pc.setType(PETSc.PC.Type.PYTHON)
            subdomains = build_subdomains(V, domain, config)
            custom_pc = OptimizedSchwarzPC(V, subdomains, A)
            pc.setPythonContext(custom_pc)
        else:
            pc.setType(config.solver.preconditioner)

        ksp.setUp()  # ← moved here, last after everything is configured

    if MPI.COMM_WORLD.rank == 0:
        print(f"\n--- Solving with {config.solver.solver_type.upper()} "
              f"(PC: {config.solver.preconditioner.upper()}) ---")


    # 6. Solve the system
    uh.x.petsc_vec.set(0.0)   # ← ADD
    ksp.solve(b, uh.x.petsc_vec)
    uh.x.scatter_forward()


    # 6. Compute L2 error
    error = uh - u_exact
    dx = ufl.Measure("dx", domain=domain)
    M = fem.form(ufl.inner(error, error) * dx)
    error_L2 = np.sqrt(MPI.COMM_WORLD.allreduce(fem.assemble_scalar(M), op=MPI.SUM))

    if MPI.COMM_WORLD.rank == 0:
        print(f"L2 error: {error_L2:.5e}")

    return uh, u_exact, error_L2