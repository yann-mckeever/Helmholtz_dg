import numpy as np
from dolfinx import mesh
from petsc4py import PETSc

def build_subdomains(V, domain, config):
    """
    Slices the domain into 4 quadrants by CELL, not by DoF.
    This guarantees that the DG element blocks remain perfectly intact.
    """
    L1 = config.mesh.L1
    L3 = config.mesh.L3
    mid_x = L1 / 2.0
    mid_y = L3 / 2.0 

    # Get the geometric midpoints of all cells in the mesh
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    cell_indices = np.arange(num_cells, dtype=np.int32)
    midpoints = mesh.compute_midpoints(domain, domain.topology.dim, cell_indices)

    dofs_Q1, dofs_Q2, dofs_Q3, dofs_Q4 = [], [], [], []

    # Iterate over cells, sorting them into quadrants based on their center point
    for cell in cell_indices:
        x, y = midpoints[cell][0], midpoints[cell][1]
        
        # Extract ALL degrees of freedom for this specific cell simultaneously
        cell_dofs = V.dofmap.cell_dofs(cell)
        
        if x <= mid_x and y <= mid_y:
            dofs_Q1.extend(cell_dofs)
        elif x > mid_x and y <= mid_y:
            dofs_Q2.extend(cell_dofs)
        elif x <= mid_x and y > mid_y:
            dofs_Q3.extend(cell_dofs)
        else:
            dofs_Q4.extend(cell_dofs)

    # Convert Python lists into PETSc Index Sets (IS)
    subdomains = []
    for dof_list in [dofs_Q1, dofs_Q2, dofs_Q3, dofs_Q4]:
        if len(dof_list) == 0:
            continue # Safely skip empty domains if the mesh is highly unusual
            
        # np.unique ensures we pass a clean, 32-bit array to PETSc
        dof_array = np.unique(dof_list).astype(np.int32)
        petsc_is = PETSc.IS().createGeneral(dof_array, comm=domain.comm)
        subdomains.append(petsc_is)

    if domain.comm.rank == 0:
        print(f"DDM: Split domain into {len(subdomains)} active subdomains.")

    return subdomains