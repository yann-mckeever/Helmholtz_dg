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

        # Normalisation : y = y / dof_counts
        # y_array = y.getArray()
        # dof_counts_array = dof_counts.getArray()
        # # Évite division par zéro (DOFs non couverts)
        # mask = np.abs(dof_counts_array) < 1e-10
        # dof_counts_array[mask] = 1.0
        # y_array[:] = y_array / dof_counts_array
        # y.resetArray()
        dof_counts.resetArray()
        dof_counts.destroy()
