from dataclasses import dataclass, field

@dataclass
class MeshConfig:
    """  
    L1 : first dimension
    L2 : second dimesion (for octagon this is the radius of center octagon, to be changed)
    L3 : third dimension
    mesh_size
    dimension
    """


    shape: str = "octagon"
    L1: float = 2.0
    L2: float = 0.05
    L3: float = 1.0
    mesh_size: float = 0.15
    dimension: int = 2

@dataclass
class PhysicsConfig:
    k: float = 15.0
    degree: int = 2
    penalty_gamma_0: float = 100.0
    penalty_gamma_1: complex = complex(-0.07, 0.01)   # use built-in complex
    penalty_beta_1: float = 20.0
    penalty_sigma: float = 1.0

@dataclass
class SolverConfig:
    solver_type: str = "direct"
    preconditioner: str = "mumps"
    use_mumps: bool = False

@dataclass
class ReferenceConfig:
    exact_solution_type: str = "Hankel"

@dataclass
class HelmholtzConfig:
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)