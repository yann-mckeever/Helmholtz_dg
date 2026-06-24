import numpy as np
from dolfinx import mesh
from petsc4py import PETSc

def build_subdomains(V, domain, config):
    L1 = config.mesh.L1
    L3 = config.mesh.L3
    mid_x = L1 / 2.0
    mid_y = L3 / 2.0
    overlap = 2 * config.mesh.mesh_size  # Chevauchement de 2 couches

    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    midpoints = mesh.compute_midpoints(domain, domain.topology.dim, cell_indices)

    dofs_Q1, dofs_Q2, dofs_Q3, dofs_Q4 = [], [], [], []

    for cell in cell_indices:
        x, y = midpoints[cell][0], midpoints[cell][1]
        cell_dofs = V.dofmap.cell_dofs(cell)

        # Quadran principal
        in_Q1 = (x <= mid_x) and (y <= mid_y)
        in_Q2 = (x > mid_x) and (y <= mid_y)
        in_Q3 = (x <= mid_x) and (y > mid_y)
        in_Q4 = (x > mid_x) and (y > mid_y)

        # Chevauchement : ajoute si proche des frontières
        near_x = abs(x - mid_x) <= overlap
        near_y = abs(y - mid_y) <= overlap

        if in_Q1 or (near_x and y <= mid_y) or (near_y and x <= mid_x):
            dofs_Q1.extend(cell_dofs)
        if in_Q2 or (near_x and y <= mid_y) or (near_y and x > mid_x):
            dofs_Q2.extend(cell_dofs)
        if in_Q3 or (near_x and y > mid_y) or (near_y and x <= mid_x):
            dofs_Q3.extend(cell_dofs)
        if in_Q4 or (near_x and y > mid_y) or (near_y and x > mid_x):
            dofs_Q4.extend(cell_dofs)

    # Crée les IS
    subdomains = []
    for dof_list in [dofs_Q1, dofs_Q2, dofs_Q3, dofs_Q4]:
        if len(dof_list) > 0:
            dof_array = np.unique(dof_list).astype(np.int32)
            petsc_is = PETSc.IS().createGeneral(dof_array, comm=domain.comm)
            subdomains.append(petsc_is)

    return subdomains