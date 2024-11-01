import numpy as np


class FitnessInterface:
    comp = None
    best_f = None
    best_x = None
    toMax = False

    def __init__(self, toMaximize=False):
        self.toMax = toMaximize
        if self.toMax:
            self.best_f = np.finfo("d").min
        else:
            self.best_f = np.finfo("d").max
        pass

    def set_comp(self, comp):
        self.comp = comp
        if self.toMax:
            self.best_f = np.finfo("d").min
        else:
            self.best_f = np.finfo("d").max
        pass

    def fitness(self, x, nlopt_func_data=None):
        f = self.comp.get_error(x)
        if (not self.toMax and f < self.best_f) or (
            self.toMax and f > self.best_f
        ):
            self.best_f = f
            self.best_x = np.array(x, copy=True)
        return f
