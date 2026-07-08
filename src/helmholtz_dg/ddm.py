import numpy as np
from dolfinx import mesh
from petsc4py import PETSc


def _factor_grid(n_subdomains):
    """Return (nx, ny) as close to a square grid as possible with nx*ny == n_subdomains."""
    n_subdomains = max(1, int(n_subdomains))
    nx = int(np.floor(np.sqrt(n_subdomains)))
    while nx > 1 and n_subdomains % nx != 0:
        nx -= 1
    ny = n_subdomains // nx
    return nx, ny

# def build_subdomains(V, domain, config):
#     L1 = config.mesh.L1
#     L3 = config.mesh.L3
#     mid_x = L1 / 2.0
#     mid_y = L3 / 2.0
#     overlap = 2 * config.mesh.mesh_size  # Chevauchement de 2 couches
#
#     num_cells = domain.topology.index_map(domain.topology.dim).size_local
#     cell_indices = np.arange(num_cells, dtype=np.int32)
#     midpoints = mesh.compute_midpoints(domain, domain.topology.dim, cell_indices)
#
#     dofs_Q1, dofs_Q2, dofs_Q3, dofs_Q4 = [], [], [], []
#
#     for cell in cell_indices:
#         x, y = midpoints[cell][0], midpoints[cell][1]
#         cell_dofs = V.dofmap.cell_dofs(cell)
#
#         # Quadran principal
#         in_Q1 = (x <= mid_x) and (y <= mid_y)
#         in_Q2 = (x > mid_x) and (y <= mid_y)
#         in_Q3 = (x <= mid_x) and (y > mid_y)
#         in_Q4 = (x > mid_x) and (y > mid_y)
#
#         # Chevauchement : ajoute si proche des frontières
#         near_x = abs(x - mid_x) <= overlap
#         near_y = abs(y - mid_y) <= overlap
#
#         if in_Q1 or (near_x and y <= mid_y) or (near_y and x <= mid_x):
#             dofs_Q1.extend(cell_dofs)
#         if in_Q2 or (near_x and y <= mid_y) or (near_y and x > mid_x):
#             dofs_Q2.extend(cell_dofs)
#         if in_Q3 or (near_x and y > mid_y) or (near_y and x <= mid_x):
#             dofs_Q3.extend(cell_dofs)
#         if in_Q4 or (near_x and y > mid_y) or (near_y and x > mid_x):
#             dofs_Q4.extend(cell_dofs)
#
#     # Crée les IS
#     subdomains = []
#     for dof_list in [dofs_Q1, dofs_Q2, dofs_Q3, dofs_Q4]:
#         if len(dof_list) > 0:
#             dof_array = np.unique(dof_list).astype(np.int32)
#             petsc_is = PETSc.IS().createGeneral(dof_array, comm=domain.comm)
#             subdomains.append(petsc_is)
#
#     return subdomains

def build_subdomains(V, domain, config, n_subdomains=None, overlap_layers=2):
    """
    Build an overlapping Cartesian nx*ny grid of subdomains covering the
    bounding box of the mesh, with a configurable total subdomain count
    and overlap width.
 
    Parameters
    ----------
    n_subdomains : int, optional
        Total number of subdomains. Defaults to config.solver.n_subdomains
        (falls back to 4 if that attribute doesn't exist, for backward
        compatibility with older configs).
    overlap_layers : int
        Overlap width expressed in multiples of config.mesh.mesh_size.
        2 layers is a reasonable default for a DG discretization; increase
        it if the local solves need more overlap to help GMRES.
 
    Notes
    -----
    Compared to the previous fixed 2x2 quadrant split, this:
      - scales to any subdomain count (not just 4),
      - keeps subdomains close to square (balanced DOF counts per block,
        rather than long thin strips), by factoring n_subdomains into the
        (nx, ny) pair closest to sqrt(n_subdomains),
      - derives the overlap band directly from the mesh size instead of
        a hardcoded constant.
    """
    if n_subdomains is None:
        n_subdomains = getattr(config.solver, "n_subdomains", 4)
 
    nx, ny = _factor_grid(n_subdomains)
 
    tdim = domain.topology.dim
    x_all = domain.geometry.x
    xmin, xmax = float(x_all[:, 0].min()), float(x_all[:, 0].max())
    ymin, ymax = float(x_all[:, 1].min()), float(x_all[:, 1].max())
 
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    overlap = overlap_layers * config.mesh.mesh_size
 
    num_cells = domain.topology.index_map(tdim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    midpoints = mesh.compute_midpoints(domain, tdim, cell_indices)
 
    dof_lists = [[] for _ in range(nx * ny)]
 
    for cell in cell_indices:
        x, y = midpoints[cell][0], midpoints[cell][1]
        cell_dofs = V.dofmap.cell_dofs(cell)
 
        # Grid block containing the cell's midpoint (clamped to valid range
        # in case of floating point rounding at the domain boundary).
        ix_center = min(max(int((x - xmin) / dx), 0), nx - 1)
        iy_center = min(max(int((y - ymin) / dy), 0), ny - 1)
 
        # A cell can belong to several neighbouring blocks if it falls
        # within the overlap band of those blocks.
        for ix in (ix_center - 1, ix_center, ix_center + 1):
            if not (0 <= ix < nx):
                continue
            x_lo = xmin + ix * dx - overlap
            x_hi = xmin + (ix + 1) * dx + overlap
            if not (x_lo <= x <= x_hi):
                continue
            for iy in (iy_center - 1, iy_center, iy_center + 1):
                if not (0 <= iy < ny):
                    continue
                y_lo = ymin + iy * dy - overlap
                y_hi = ymin + (iy + 1) * dy + overlap
                if not (y_lo <= y <= y_hi):
                    continue
                dof_lists[iy * nx + ix].extend(cell_dofs)
 
    subdomains = []
    for k, dof_list in enumerate(dof_lists):
        if len(dof_list) == 0:
            # Can happen for a very coarse mesh with too many requested
            # subdomains; skip empty blocks rather than build an empty IS.
            continue
        dof_array = np.unique(dof_list).astype(np.int32)
        petsc_is = PETSc.IS().createGeneral(dof_array, comm=domain.comm)
        subdomains.append(petsc_is)
 
    if domain.comm.rank == 0:
        sizes = [isub.getSize() for isub in subdomains]
        print(f"[build_subdomains] grid {nx}x{ny} -> {len(subdomains)} subdomains, "
              f"DOF counts: min={min(sizes)}, max={max(sizes)}, overlap={overlap:.4g}")
 
    return subdomains
