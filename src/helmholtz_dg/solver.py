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
from .preconditioner import AdditiveSchwarzPC
from .preconditioner import TwoLevelASM

def solve_problem(config: HelmholtzConfig):
    """
    Solve the DG Helmholtz problem with the given configuration.
    Returns:
        uh: dolfinx.fem.Function – computed solution
        u_exact: dolfinx.fem.Function – exact solution (interpolated)
        error_L2: float – L2 error norm
    """
    # 1. Create mesh
    mesh_data = create_mesh(config)
    domain = mesh_data[0]
    facet_tags = mesh_data[2]

    # 2. Build forms and function space
    V, u, v, a, L, u_exact = build_problem(domain, config, facet_tags)

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
        ksp.setType("gmres")
        ksp.setTolerances(rtol=1e-8, atol=1e-14, max_it=500)  # Plus strict 
        ksp.setMonitor(lambda ksp, its, rnorm: print(f"GMRES: it={its}, residual={rnorm:.2e}"))  # Suivi en temps réel
        ksp.setGMRESRestart(100)
        pc = ksp.getPC()

        if config.solver.preconditioner == "custom_asm":
            pc.setType(PETSc.PC.Type.PYTHON)
            # pc.setType(PETSc.PC.Type.ASM)
            # pc.setASMType(PETSc.PC.ASMType.RESTRICT)   # RAS — usually better than pure additive
            # pc.setGASMOverlap(2)                        # overlap layers
            # sub_ksps = pc.getASMSubKSP()
            # for sk in sub_ksps:
            #     sk.setType("preonly")
            #     sk.getPC().setType("lu")
            #     sk.getPC().setFactorSolverType("mumps")
            subdomains = build_subdomains(V, domain, config, n_subdomains=8)
            custom_pc = AdditiveSchwarzPC(V, subdomains, A)
            # custom_pc = TwoLevelASM(V, subdomains, A)
            pc.setPythonContext(custom_pc)
        else:
            pc.setType(config.solver.preconditioner) 
        ksp.setUp()
        
       

        # if config.solver.preconditioner == "custom_asm":
        #     pc.setType(PETSc.PC.Type.ASM)
        #
        #     coarse_ksp = pc.getASMCoarseKSP()  # ← Cela active automatiquement le coarse operator
        #
        #     # Configure le solveur grossier
        #     coarse_ksp.setType("preonly")
        #     coarse_pc = coarse_ksp.getPC()
        #     coarse_pc.setType("lu")
        #     coarse_pc.setFactorSolverType("mumps")
        #
        #     # Configure les sous-solveurs fins
        #     sub_ksps = pc.getASMSubKSP()
        #     for sub_ksp in sub_ksps:
        #         sub_ksp.setType("preonly")
        #         sub_pc = sub_ksp.getPC()
        #         sub_pc.setType("lu")
        #         sub_pc.setFactorSolverType("mumps")
        # else:
        #     pc.setType(config.solver.preconditioner)
        # ksp.setUp()

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

    #if MPI.COMM_WORLD.rank == 0:
    print(f"L2 error: {error_L2:.5e}")

    print("Converged reason:", ksp.getConvergedReason(), "iterations:", ksp.getIterationNumber())
    return uh, u_exact, error_L2
