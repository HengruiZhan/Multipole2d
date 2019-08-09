import numpy as np
import scipy.constants as sc
from scipy.special import sph_harm


class Multipole():
    """The Multipole is written in vetorized codes"""

    def __init__(self, grid, lmax, dr, center=(0.0, 0.0)):

        self.g = grid
        self.n_moments = lmax+1
        self.dr_mp = dr
        self.center = center

        # compute the bins
        # this computation method is not correct
        r_max = max(abs(self.g.rlim[0] - center[0]), abs(self.g.rlim[1] -
                                                         center[0]))
        z_max = max(abs(self.g.zlim[0] - center[1]), abs(self.g.zlim[1] -
                                                         center[1]))

        dmax = np.sqrt(r_max**2 + z_max**2)

        self.n_bins = int(dmax/dr)

        # bin boundaries
        self.r_bin = np.linspace(0.0, dmax, self.n_bins)

        # storage for the inner and outer multipole moment functions
        # we'll index the list by multipole moment l

        self.m_r = []
        self.m_i = []
        for l in range(self.n_moments):
            self.m_r.append(np.zeros((self.n_bins), dtype=np.complex128))
            self.m_i.append(np.zeros((self.n_bins), dtype=np.complex128))

    def compute_harmonics(self, l, r, z):
        # r and z are all array
        # modified!!
        radius = np.sqrt((r - self.center[0])**2 +
                         (z - self.center[1])**2)
        # tan(theta) = r/z
        theta = np.arctan2(r, z)

        Y_lm = sph_harm(0, l, 0.0, theta)
        R_lm = np.sqrt(4*np.pi/(2*l + 1)) * radius**l * Y_lm
        I_lm = np.nan_to_num(np.sqrt(4*np.pi/(2*l + 1)) * Y_lm / radius**(l+1))

        return R_lm, I_lm

    def compute_expansion(self, rho):
        # rho is density that lives on a grid self.g
        radius = np.sqrt((self.g.r - self.center[0])**2 +
                         (self.g.z - self.center[1])**2)

        m_zone = rho * self.g.vol
        # loop over the multipole moments, l (m = 0 here)

        for l in range(self.n_moments):

            # compute Y_l^m (note: we use theta as the polar
            # angle, scipy is opposite)
            R_lm, I_lm = self.compute_harmonics(l, self.g.r, self.g.z)

            # add to the all of the appropriate inner or outer
            # moment functions
            for i in range(self.n_bins):
                imask = radius <= self.r_bin[i]
                omask = radius > self.r_bin[i]
                self.m_r[l][i] += np.sum(R_lm[imask] * m_zone[imask])
                self.m_i[l][i] += np.sum(I_lm[omask] * m_zone[omask])

    def sample_mtilde(self, l, r):
        # use digitize to modify
        # this returns the result of Eq. 19

        # we need to find which be we are in

        mu_m = np.argwhere(self.r_bin <= r)[-1][0]
        mu_p = np.argwhere(self.r_bin > r)[0][0]

        assert mu_p == mu_m + 1

        mtilde_r = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_r[l][mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_r[l][mu_m]

        mtilde_i = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_i[l][mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_i[l][mu_m]
        return mtilde_r, mtilde_i

    def phi(self, r, z):
        # return Phi(r), using Eq. 20
        # evaluated at the face of the cell

        radius = np.sqrt((r - self.center[0])**2 +
                         (z - self.center[1])**2)
        phi_zone = 0.0
        for l in range(self.n_moments):
            mtilde_r, mtilde_i = self.sample_mtilde(l, radius)
            # calculate the average of the solid harmonic function of all
            # the surface
            # solid harmonic function at r-dr/2 surface
            R_lm_r_minus, I_lm_r_minus =\
                self.compute_harmonics(l, r-self.g.dr/2, z)
            # solid harmonic function at r+dr/2 surface
            R_lm_r_plus, I_lm_r_plus =\
                self.compute_harmonics(l, r+self.g.dr/2, z)
            # solid harmonic function at r-dz/2 surface
            R_lm_z_minus, I_lm_z_minus =\
                self.compute_harmonics(l, r, z-self.g.dz/2)
            # solid harmonic function at r+dz/2 surface
            R_lm_z_plus, I_lm_z_plus =\
                self.compute_harmonics(l, r, z+self.g.dz/2)
            # average harmonic function of all the surface
            R_lm = 1/4*(R_lm_r_minus+R_lm_r_plus+R_lm_z_minus+R_lm_z_plus)
            I_lm = 1/4*(I_lm_r_minus+I_lm_r_plus+I_lm_z_minus+I_lm_z_plus)
            # calculate Eq. 20
            phi_zone += sc.G * (mtilde_r * np.conj(I_lm) +
                                np.conj(mtilde_i) * R_lm)

        return -np.real(phi_zone)
