import numpy as np


class Grid():
    def __init__(self, nr, nz, rlim=(0.0, 1.0), zlim=(-1.0, 1.0)):
        """an axisymmetric grid"""

        self.nr = nr
        self.nz = nz

        self.rlim = rlim
        self.zlim = zlim

        self.dr = (self.rlim[1] - self.rlim[0])/nr
        self.dz = (self.zlim[1] - self.zlim[0])/nz

        self.r = (np.arange(self.nr) + 0.5)*self.dr + self.rlim[0]

        self.z = (np.arange(self.nz) + 0.5)*self.dz + self.zlim[0]

        self.r2d, self.z2d = np.meshgrid(self.r, self.z, indexing='ij')

        self.vol = np.pi*2.0*self.r2d*self.dr*self.dz

    def scratch_array(self):
        return np.zeros((self.nr, self.nz), dtype=np.float64)
