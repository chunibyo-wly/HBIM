import numpy as np

from semregpy.component.interface import *


class OfficeComponent(ComponentInterface):
    """Parametric Door component (x, y, z, r_z) implements ComponentInterface.
    Attributes
    ----------
    fit : Fitness
      The associated fitness and function.
    dim : int
      The dimensions of independent params.
    params : dict
      (private) The parameters to optimize
    """

    type = "OfficeFurniture"
    dim = 3
    stage = 2  # 1 = x,y,rz 2 = z,r_z

    fit = None
    params = {}
    resolution = 0.01  # 1cm
    txt = ""

    def __init__(
        self,
        params: dict = {
            "min": [0, 0, 0],
            "max": [0, 0, 0],
            "c": [0.5, 0.5, 0],
            "rz": 0,
            "hv_weights": [0.5, 0.5],
            "rz_display": 0,
            "ignore_z": 0.3,
        },
    ):
        self.params = params
        self.stage = 1

    def set_stage(self, v):
        self.stage = v
        if v == 1:
            self.dim = 3
        else:
            self.dim = 2

    def get_dof(self) -> int:
        """Return degree of freedom (DoF)."""
        return self.dim

    def decode_x(self, x: np.array) -> dict:
        """Decode config from array.
        Parameters
        ----------
        x : np.array
          In the format [x%, y%] or [z%, rz%];
        """
        if self.stage == 1:
            for axis in [0, 1]:
                v = x[axis]
                if v > 1.0:
                    v = 1.0
                elif v < 0:
                    v = 0
                self.params["c"][axis] = self.params["min"][axis] + v * (
                    self.params["max"][axis] - self.params["min"][axis]
                )
            axis = 2
            self.params["c"][axis] = self.params["c_mesh"][axis]
            self.params["rz"] = np.pi * (2 * x[2] - 1)
        else:
            for axis in [0, 1]:
                self.params["c"][axis] = self.params["best_c"][axis]
            for axis in [0]:
                v = x[axis]
                if v > 1.0:
                    v = 1.0
                elif v < 0:
                    v = 0
                self.params["c"][2] = (v + 0.5) * self.params["c_mesh"][2]
            self.params["rz"] = np.pi * (2 * x[1] - 1)
        return self.params
