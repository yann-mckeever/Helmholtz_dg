from .config import HelmholtzConfig
from mpi4py import MPI
import gmsh
from dolfinx.io import gmsh as gmsh_io
import numpy as np

from dolfinx.mesh import create_cell_partitioner, GhostMode

def create_mesh(config: HelmholtzConfig):
    comm = MPI.COMM_WORLD
    gmsh.initialize()
    gmsh.model.add("domain")

    if comm.rank == 0:
        if config.mesh.shape=="rectangle":
            L1=config.mesh.L1 #first side length
            L3=config.mesh.L3 #second side length
            h=config.mesh.mesh_size

            p1 = gmsh.model.geo.addPoint(0, 0, 0, h)
            p2 = gmsh.model.geo.addPoint(L1, 0, 0, h)
            p3 = gmsh.model.geo.addPoint(L1, L3, 0, h)
            p4 = gmsh.model.geo.addPoint(0, L3, 0, h)

            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)

            loop = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            surface = gmsh.model.geo.addPlaneSurface([loop])
            
            gmsh.model.geo.synchronize()

            # Assign Physical Groups
            gmsh.model.addPhysicalGroup(2, [surface], 1)             # Volume

            if config.reference.exact_solution_type=="Case_1":
                gmsh.model.addPhysicalGroup(1, [l3,l4], 1)
                gmsh.model.addPhysicalGroup(1, [l1, l2], 2)
            elif config.reference.exact_solution_type == "PlaneWave":
                # Tag all outer boundaries as 1
                gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], 1)

            else:
                gmsh.model.addPhysicalGroup(1, [l1, l2, l3, l4], 1)      # All boundaries


        
        if config.mesh.shape=="octagon":
            # Build Octagon Points & Lines
            pts = [gmsh.model.geo.addPoint(config.mesh.L1*np.cos(i*np.pi/4), config.mesh.L1*np.sin(i*np.pi/4), 0, config.mesh.mesh_size) for i in range(8)]
            lines = [gmsh.model.geo.addLine(pts[i], pts[(i+1)%8]) for i in range(8)]
            out_loop = gmsh.model.geo.addCurveLoop(lines)
            
            # Build Center Micro-Hole
            pc = gmsh.model.geo.addPoint(0, 0, 0, config.mesh.L2/4)
            p1 = gmsh.model.geo.addPoint(config.mesh.L2, 0, 0, config.mesh.L2/4)
            p2 = gmsh.model.geo.addPoint(-config.mesh.L2, 0, 0, config.mesh.L2/4)
            c1 = gmsh.model.geo.addCircleArc(p1, pc, p2)
            c2 = gmsh.model.geo.addCircleArc(p2, pc, p1)
            in_loop = gmsh.model.geo.addCurveLoop([c1, c2])

            # Generate Surface
            surface = gmsh.model.geo.addPlaneSurface([out_loop, in_loop])
            gmsh.model.geo.synchronize()

            # Assign Physical Groups (Required for FEniCSx to read boundaries)
            gmsh.model.addPhysicalGroup(2, [surface], 1)           # Volume
            gmsh.model.addPhysicalGroup(1, lines + [c1, c2], 1)    # All boundaries


        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)
    partitioner = create_cell_partitioner(GhostMode.shared_facet, max_facet_to_cell_links=2)
    mesh_data = gmsh_io.model_to_mesh(
        gmsh.model, 
        MPI.COMM_WORLD, 
        0, 
        gdim=2, 
        partitioner=partitioner
    )
    gmsh.finalize()

    return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags

