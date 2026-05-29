from .config import MeshConfig

from mpi4py import MPI
import gmsh
from dolfinx.io import gmsh as gmsh_io
import numpy as np

def create_mesh(config: MeshConfig):
    gmsh.initialize()
    gmsh.model.add("domain")

    if config.shape=="octagon":
        # Build Octagon Points & Lines
        pts = [gmsh.model.geo.addPoint(config.L1*np.cos(i*np.pi/4), config.L1*np.sin(i*np.pi/4), 0, config.mesh_size) for i in range(8)]
        lines = [gmsh.model.geo.addLine(pts[i], pts[(i+1)%8]) for i in range(8)]
        out_loop = gmsh.model.geo.addCurveLoop(lines)
        
        # Build Center Micro-Hole
        pc = gmsh.model.geo.addPoint(0, 0, 0, config.L2/4)
        p1 = gmsh.model.geo.addPoint(config.L2, 0, 0, config.L2/4)
        p2 = gmsh.model.geo.addPoint(-config.L2, 0, 0, config.L2/4)
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
        mesh_data = gmsh_io.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=2)
        gmsh.finalize()

        return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags

