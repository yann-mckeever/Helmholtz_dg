# main.py (top level, outside src/)
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from helmholtz_dg.config import HelmholtzConfig
from helmholtz_dg.solver import solve_problem
from helmholtz_dg.output import save_solution_to_xdmf, save_error_to_xdmf, export_animation

def main():
    # --- Setup configuration ---
    cfg = HelmholtzConfig()
    #---Domain/Mesh config---
    cfg.mesh.shape = "rectangle"
    #cfg.mesh.shape = "octagon"
    cfg.mesh.L1 = 1.0
    cfg.mesh.L2 = 0.005
    cfg.mesh.L3 = 1.0
    cfg.mesh.mesh_size = 0.01
    #---Physical parameters config--- (cf DOI 10.1137/080737538)
    cfg.physics.k = 0.1
    cfg.physics.degree = 2
    cfg.physics.penalty_gamma_0 = 100.0
    cfg.physics.penalty_i_gamma_1 = complex(-0.07, 0.01)
    cfg.physics.penalty_beta_1 = 20.0
    cfg.physics.penalty_sigma = 1.0

    #---Solver config---
    cfg.solver.solver_type = "gmres" #"direct" #"gmres"
    cfg.solver.preconditioner = "custom_asm"  # 'ilu', 'jacobi', or 'none' "custom_asm"

    cfg.reference.exact_solution_type = "PlaneWave"

    # --- Solve ---
    uh, u_exact, error_L2 = solve_problem(cfg)

    # --- Output for ParaView ---
    domain = uh.function_space.mesh

    # 1. Static exports
    save_solution_to_xdmf(uh, domain, "helmholtz_solution")
    save_solution_to_xdmf(u_exact, domain, "helmholtz_exact")
    save_error_to_xdmf(uh, u_exact, domain, "helmholtz_error")

    # 2. Animated DG Export
    #export_animation(uh, domain, cfg.physics.k, "wave_animation.bp")

if __name__ == "__main__":
    main()