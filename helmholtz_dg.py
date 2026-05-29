import numpy as np
import ufl
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import mesh as dmesh, fem, io
from dolfinx.io import VTXWriter 
from dolfinx.fem.petsc import LinearProblem
import gmsh
from scipy.special import hankel1

# ====================================================================
# MODULAR DG HELMHOLTZ SOLVER (FEniCSx)
# 2D Octagonal Domain | Center Emitter | Imaginary IPDG Penalty
# ====================================================================

# --- 1. User Parameters ---
k = 15.0              # Wavenumber
degree = 2            # Polynomial degree (DG)
penalty_beta = 20.0   # Imaginary penalty parameter
R = 2.0               # Octagon outer radius
r0 = 0.05             # Inner emitter radius

# --- 2. Mesh Generation via Gmsh ---
gmsh.initialize()
gmsh.model.add("octagon_emitter")

# Build Octagon Points & Lines
pts = [gmsh.model.geo.addPoint(R*np.cos(i*np.pi/4), R*np.sin(i*np.pi/4), 0, 0.15) for i in range(8)]
lines = [gmsh.model.geo.addLine(pts[i], pts[(i+1)%8]) for i in range(8)]
out_loop = gmsh.model.geo.addCurveLoop(lines)

# Build Center Micro-Hole
pc = gmsh.model.geo.addPoint(0, 0, 0, r0/4)
p1 = gmsh.model.geo.addPoint(r0, 0, 0, r0/4)
p2 = gmsh.model.geo.addPoint(-r0, 0, 0, r0/4)
c1 = gmsh.model.geo.addCircleArc(p1, pc, p2)
c2 = gmsh.model.geo.addCircleArc(p2, pc, p1)
in_loop = gmsh.model.geo.addCurveLoop([c1, c2])

# Generate Surface
surface = gmsh.model.geo.addPlaneSurface([out_loop, in_loop])
gmsh.model.geo.synchronize()

# Assign Physical Groups (Required for FEniCSx to read boundaries)
gmsh.model.addPhysicalGroup(2, [surface], 1)           # Volume
gmsh.model.addPhysicalGroup(1, lines + [c1, c2], 1)    # All boundaries

gmsh.model.mesh.generate(2)
from dolfinx.io import gmsh as gmsh_io
mesh_data = gmsh_io.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
domain = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags
gmsh.finalize()

# --- 3. Function Space & DG Setup ---
# Initialize the complex Discontinuous Galerkin space
V = fem.functionspace(domain, ("DG", degree))
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Geometric and mesh size properties
n = ufl.FacetNormal(domain)
h = ufl.CellDiameter(domain)
h_avg = (h('+') + h('-')) / 2.0

# Complex Penalty Parameters (1j is the imaginary unit in Python)
alpha_int = 1j * penalty_beta * (degree**2) / h_avg
alpha_bnd = 1j * penalty_beta * (degree**2) / h

# --- 4. Exact Solution ---
u_exact = fem.Function(V)

def exact_solution(x_coords):
    r = np.sqrt(x_coords[0]**2 + x_coords[1]**2)
    return hankel1(0, k * r)

# Interpolate the numpy Hankel function into the FEniCS space
u_exact.interpolate(exact_solution)

# --- 5. DG Weak Formulation (Imaginary IPDG) ---
# Volume integral
a = ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx - k**2 * ufl.inner(u, v) * ufl.dx

# Skeleton (Interior element faces)
a += - ufl.inner(ufl.avg(ufl.grad(u)), ufl.jump(v, n)) * ufl.dS \
     - ufl.inner(ufl.jump(u, n), ufl.avg(ufl.grad(v))) * ufl.dS \
     + alpha_int * ufl.inner(ufl.jump(u, n), ufl.jump(v, n)) * ufl.dS

# Outer & Inner Boundaries (Nitsche's Method for Dirichlet)
a += - ufl.inner(ufl.dot(ufl.grad(u), n), v) * ufl.ds \
     - ufl.inner(u, ufl.dot(ufl.grad(v), n)) * ufl.ds \
     + alpha_bnd * ufl.inner(u, v) * ufl.ds

# Right Hand Side (Imposing exact solution on the boundaries)
L = - ufl.inner(u_exact, ufl.dot(ufl.grad(v), n)) * ufl.ds \
    + alpha_bnd * ufl.inner(u_exact, v) * ufl.ds

# --- 6. Matrix Assembly & Solving ---
print(f"Solving system with degree P{degree} and k={k}...")

# 1. Compile the Unified Form Language (UFL) forms
a_form = fem.form(a)
L_form = fem.form(L)

# 2. Assemble the massive sparse matrix A
A = fem.petsc.assemble_matrix(a_form)
A.assemble()

# 3. Assemble the right-hand side vector b
b = fem.petsc.assemble_vector(L_form)
b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)

# 4. Create the empty solution function
uh = fem.Function(V)

# 5. Build the explicit PETSc KSP (Krylov Subspace) Solver
ksp = PETSc.KSP().create(domain.comm)
ksp.setOperators(A)
ksp.setType(PETSc.KSP.Type.PREONLY)          # Force a direct solve for now
ksp.getPC().setType(PETSc.PC.Type.LU)        # Use LU Decomposition
ksp.getPC().setFactorSolverType("mumps")     # Use the MUMPS backend

# 6. Solve the system
ksp.solve(b, uh.x.petsc_vec)
uh.x.scatter_forward() # Update parallel ghost values

# --- 7. Error Analysis ---
error_form = fem.form(ufl.inner(uh - u_exact, uh - u_exact) * ufl.dx)
error_local = fem.assemble_scalar(error_form)
L2_error = np.sqrt(np.real(domain.comm.allreduce(error_local, op=MPI.SUM)))

print(f"=======================================")
print(f" L2 Error against Hankel: {L2_error:.4e}")
print(f"=======================================")

# --- 7b. Local Error Visualization (A Priori / A Posteriori) ---
print("Calculating local error indicators...")

# Create a piecewise constant space (DG-0) to hold one value per element
V_err = fem.functionspace(domain, ("DG", 0))

# 1. Realized Error (Realization of the A Priori bound)
# Computes the exact absolute error magnitude in each cell
# FIXED: Removed the () from interpolation_points
expr_actual = fem.Expression(ufl.sqrt(ufl.inner(uh - u_exact, uh - u_exact)), V_err.element.interpolation_points)
err_actual = fem.Function(V_err)
err_actual.name = "Realized_Exact_Error"
err_actual.interpolate(expr_actual)

# 2. A Posteriori Error Indicator (Internal Element Residual)
# Formula: R_K = | \Delta u_h + k^2 u_h | (Since f = 0)
# This requires NO knowledge of the exact solution.
residual = ufl.div(ufl.grad(uh)) + k**2 * uh
# FIXED: Removed the () from interpolation_points
expr_post = fem.Expression(ufl.sqrt(ufl.inner(residual, residual)), V_err.element.interpolation_points)
err_post = fem.Function(V_err)
err_post.name = "A_Posteriori_Residual"
err_post.interpolate(expr_post)

# Save these specific static fields to a separate file so they don't mix with the time animation
with VTXWriter(domain.comm, "error_analysis.bp", [err_actual, err_post], engine="BP4") as vtx_err:
    vtx_err.write(0.0)

print("Saved static error fields to 'error_analysis.bp'.")

# --- 8. Export for Animation (ParaView) ---

print("Generating Animation Frames...")

# Create a clean, real-valued container to hold each frame of the video
uh_animated = fem.Function(V)
uh_animated.name = "Acoustic_Wave_Evolution"

# Time-harmonic animation parameters
nFrames = 60
omega = k * 1.0  # Wavenumber * Wave Speed (assuming speed = 1.0)
dt = (2 * np.pi / omega) / (nFrames / 2.0)

# Open the VTX writer and inject frames one by one
with VTXWriter(domain.comm, "helmholtz_dg.bp", [uh_animated], engine="BP4") as vtx:
    for i in range(nFrames):
        t = i * dt
        
        # Calculate the physical wave height at this exact microsecond
        # Formula: Real( u * e^{-i * omega * t} )
        current_wave_state = np.real(uh.x.array * np.exp(-1j * omega * t))
        
        # Update our container and write it to the file
        uh_animated.x.array[:] = current_wave_state
        vtx.write(t)

print("Simulation complete! Open the 'helmholtz_dg.bp' folder in ParaView.")