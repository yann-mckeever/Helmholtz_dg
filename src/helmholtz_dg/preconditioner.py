from petsc4py import PETSc

class OptimizedSchwarzPC:
    def __init__(self, V, subdomains, global_matrix):
        self.V = V
        self.subdomains = subdomains
        self.global_matrix = global_matrix
        self.local_ksp_solvers = []

    def setUp(self, pc):
        print(f"[OptimizedSchwarzPC] setUp called, building {len(self.subdomains)} subdomain solvers")
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

            self.local_ksp_solvers.append(ksp)

    def apply(self, pc, x, y):
        y.set(0.0)
        for ksp, isub in zip(self.local_ksp_solvers, self.subdomains):
            vec_local_rhs = x.getSubVector(isub)
            vec_local_sol = vec_local_rhs.duplicate()
            ksp.solve(vec_local_rhs, vec_local_sol)
            x.restoreSubVector(isub, vec_local_rhs)  
            y.setValues(isub.array, vec_local_sol.array, PETSc.InsertMode.INSERT_VALUES) 
            vec_local_sol.destroy()
        y.assemblyBegin()
        y.assemblyEnd()
