from petsc4py import PETSc
import numpy as np

class AdditiveSchwarzPC:
    def __init__(self, V, subdomains, global_matrix):
        self.V = V
        self.subdomains = subdomains
        self.global_matrix = global_matrix
        self.local_ksp_solvers = []

    def setUp(self, pc):
        print(f"[AdditiveSchwarzPC] Building {len(self.subdomains)} subdomain solvers")
        for i, isub in enumerate(self.subdomains):
            print(f"  Subdomain {i}: {isub.getSize()} DOFs")  # Vérifie la taille
            sub_mat = self.global_matrix.createSubMatrix(isub, isub)
            print(f"  Submatrix {i}: {sub_mat.getSize()}")   # Vérifie la matrice
        
        
        
        comm = self.V.mesh.comm

        for isub in self.subdomains:
            sub_mat = self.global_matrix.createSubMatrix(isub, isub)

            ksp = PETSc.KSP().create(comm=comm)
            ksp.setOperators(sub_mat)
            ksp.setType("preonly")
            ksp_pc = ksp.getPC()
            ksp_pc.setType("lu")
            ksp_pc.setFactorSolverType("mumps")
            ksp.setUp()


            """ ksp.setType("gmres")
            ksp.setTolerances(max_it=5)
            ksp_pc = ksp.getPC()
            ksp_pc.setType("ilu")
            ksp.setUp() """

            self.local_ksp_solvers.append(ksp)

    def apply(self, pc, x, y):
        y.set(0.0)
        dof_counts = y.duplicate()
        dof_counts.set(0.0)

        # Compte les contributions par DOF
        for isub in self.subdomains:
            ones = PETSc.Vec().createWithArray(
                np.ones(isub.getSize(), dtype=PETSc.ScalarType),
                comm=x.comm
            )
            dof_counts.setValues(isub.array, ones.getArray(), PETSc.InsertMode.ADD_VALUES)
            ones.destroy()

        dof_counts.assemblyBegin()
        dof_counts.assemblyEnd()

        # Applique ASM
        for ksp, isub in zip(self.local_ksp_solvers, self.subdomains):
            vec_local_rhs = x.getSubVector(isub)
            vec_local_sol = vec_local_rhs.duplicate()
            ksp.solve(vec_local_rhs, vec_local_sol)
            x.restoreSubVector(isub, vec_local_rhs)
            y.setValues(isub.array, vec_local_sol.array, PETSc.InsertMode.ADD_VALUES)
            vec_local_sol.destroy()

        y.assemblyBegin()
        y.assemblyEnd()
        y.pointwiseDivide(y, dof_counts)   # ← missing line
        dof_counts.resetArray()
        dof_counts.destroy()




class TwoLevelASM:
    """
    Two-level Additive Schwarz preconditioner (PCPython context).

    Level 1 (fine): overlapping subdomain solves, exact local LU.
    Level 2 (coarse): Nicolaides / partition-of-unity coarse space.
        - One coarse basis vector per subdomain: the subdomain's
          partition-of-unity function w_i (1 on DOFs owned only by
          subdomain i, 1/multiplicity on overlap DOFs, 0 elsewhere).
        - R0 = [w_1 | w_2 | ... | w_n] is the (n_dofs x n_subdomains)
          prolongation from coarse to fine space.
        - A0 = R0^T A R0 is the Galerkin coarse operator, solved by LU.

    M^{-1} x = R0 A0^{-1} R0^T x  +  sum_i  R_i^T A_i^{-1} R_i x

    This is the classical Bramble-Pasciak-Schatz two-level additive
    Schwarz correction: the coarse term removes the dependence of the
    GMRES iteration count on the number of subdomains that plain
    one-level ASM suffers from.
    """

    def __init__(self, V, subdomains, global_matrix):
        self.V = V
        self.subdomains = subdomains
        self.A = global_matrix

        self.local_ksp_solvers = []
        self.pou = []           # partition-of-unity weight vector per subdomain
        self.R0 = None          # coarse basis matrix (n_dofs x n_subdomains)
        self.A0 = None          # coarse Galerkin operator
        self.coarse_ksp = None

    # ------------------------------------------------------------------
    def setUp(self, pc):
        comm = self.V.mesh.comm
        n_dofs = self.A.getSize()[0]
        n_sub = len(self.subdomains)

        if n_sub == 0:
            raise RuntimeError("TwoLevelASM: no subdomains were provided.")

        # --- multiplicity of each dof across the overlapping subdomains ---
        mult = self.A.createVecRight()
        mult.set(0.0)
        for isub in self.subdomains:
            ones = np.ones(isub.getSize(), dtype=PETSc.ScalarType)
            mult.setValues(isub.array, ones, addv=PETSc.InsertMode.ADD_VALUES)
        mult.assemblyBegin()
        mult.assemblyEnd()
        # avoid division by zero for dofs that belong to no subdomain
        mult.array[mult.array == 0.0] = 1.0
        inv_mult = mult.copy()
        inv_mult.reciprocal()

        # --- fine level: local direct solvers + POU weight per subdomain ---
        for isub in self.subdomains:
            sub_mat = self.A.createSubMatrix(isub, isub)

            ksp = PETSc.KSP().create(comm=comm)
            ksp.setOperators(sub_mat)
            ksp.setType("preonly")
            ksp_pc = ksp.getPC()
            ksp_pc.setType("lu")
            try:
                ksp_pc.setFactorSolverType("mumps")
            except PETSc.Error:
                pass  # fall back to PETSc's built-in LU if MUMPS isn't available
            ksp.setUp()
            self.local_ksp_solvers.append(ksp)

            sub_vec = inv_mult.getSubVector(isub)
            w = sub_vec.copy()
            inv_mult.restoreSubVector(isub, sub_vec)
            self.pou.append(w)

        # --- coarse level: Nicolaides basis R0 and Galerkin operator A0 ---
        R0 = PETSc.Mat().createAIJ([n_dofs, n_sub], comm=comm)
        R0.setUp()
        for j, (isub, w) in enumerate(zip(self.subdomains, self.pou)):
            rows = isub.array
            cols = np.full(1, j, dtype=PETSc.IntType)
            R0.setValues(rows, cols, w.array, addv=PETSc.InsertMode.ADD_VALUES)
        R0.assemblyBegin()
        R0.assemblyEnd()
        self.R0 = R0

        AR0 = self.A.matMult(R0)
        self.A0 = R0.transposeMatMult(AR0)
        AR0.destroy()

        self.coarse_ksp = PETSc.KSP().create(comm=comm)
        self.coarse_ksp.setOperators(self.A0)
        self.coarse_ksp.setType("preonly")
        coarse_pc = self.coarse_ksp.getPC()
        coarse_pc.setType("lu")
        try:
            coarse_pc.setFactorSolverType("mumps")
        except PETSc.Error:
            pass
        self.coarse_ksp.setUp()

        mult.destroy()
        inv_mult.destroy()

        if comm.rank == 0:
            print(f"[TwoLevelASM] {n_sub} subdomains, coarse space dim = {n_sub}")

    # ------------------------------------------------------------------
    def apply(self, pc, x, y):
        y.set(0.0)

        # ---- Level 1: overlapping subdomain corrections ----
        for ksp, isub in zip(self.local_ksp_solvers, self.subdomains):
            x_loc = x.getSubVector(isub)
            y_loc = x_loc.duplicate()
            ksp.solve(x_loc, y_loc)
            x.restoreSubVector(isub, x_loc)
            y.setValues(isub.array, y_loc.array, addv=PETSc.InsertMode.ADD_VALUES)
            y_loc.destroy()
        y.assemblyBegin()
        y.assemblyEnd()

        # ---- Level 2: coarse (Nicolaides) correction ----
        r0 = self.R0.createVecRight()
        self.R0.multTranspose(x, r0)      # r0 = R0^T x
        e0 = r0.duplicate()
        self.coarse_ksp.solve(r0, e0)     # e0 = A0^{-1} R0^T x
        y_coarse = y.duplicate()
        self.R0.mult(e0, y_coarse)        # y_coarse = R0 e0
        y.axpy(1.0, y_coarse)

        # print("norm y_coarse:", y_coarse.norm(), "norm y_fine:", y.norm())
        r0.destroy()
        e0.destroy()
        y_coarse.destroy()

