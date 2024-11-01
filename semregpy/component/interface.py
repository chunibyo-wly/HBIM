import numpy as np

from semregpy.fitness.interface import *


class ComponentInterface:
    def __init__(self, params: dict) -> None:
        """Set up a parametric Semantic Object. Inputs fitness reference data, construction params."""
        pass

    def get_dof(self) -> int:
        """Return degree of freedom (DoF)."""
        pass

    def get_fitness(self) -> FitnessInterface:
        pass

    def decode_x(self, x: np.array) -> dict:
        """Decode config from array."""
        pass

    def get_error(self, x: np.array) -> np.double:
        """Returns error from the setting x."""
        pass

    def get_geometry(self, x: np.array) -> dict:
        """Returns the geometry of setting x."""
        pass
