from numba import jit
import numpy as np
import scipy.constants as sc


@jit
def calcLegPolyL(l, x):
    # Calculate the Legendre polynomials. We use a stable recurrence relation:
    # (l+1) P_{l+1}(x) = (2l+1) x P_l(x) - l P_{l-1}(x).
    # This uses initial conditions:
    # P_0(x) = 1
    # P_1(x) = x
    if (l == 0):
        return 1.0
    elif(l == 1):
        return x
    else:
        legPolyL2 = 1.0
        legPolyL1 = x
        legPolyL = 0.0
        for n in range(2, l+1):
            legPolyL = ((2*n - 1) * x * legPolyL1 - (n-1) * legPolyL2) / n
            legPolyL2 = legPolyL1
            legPolyL1 = legPolyL
        return legPolyL


class Multipole2d():
    '''The Multipole Expansion in 2d case'''

    def __init__(self, grid, lmax, dr, density, center=(0.0, 0.0)):

        self.g = grid
        self.lmax = lmax
        self.dr = dr
        self.density = density
        self.center = center

        self.r = self.g.r2d - self.center[0]
        self.z = self.g.z2d - self.center[1]
        self.radius = np.sqrt(self.r**2 + self.z**2)
        self.cosTheta = self.z/self.radius

        self.m = self.density * self.g.vol

        self.phi = self.g.scratch_array()

        # compute the bins
        r_max = max(abs(self.g.rlim[0] - center[0]), abs(self.g.rlim[1] -
                                                         center[0]))
        z_max = max(abs(self.g.zlim[0] - center[1]), abs(self.g.zlim[1] -
                                                         center[1]))

        # radii of concentric spheres
        dmax = np.sqrt(r_max**2 + z_max**2)

        self.n_bins = int(dmax/dr)

        # r_bin do not need to start at 0, but need to include every grid cell
        # center
        self.r_bin = np.linspace(0, dmax, self.n_bins)
        # self.r_bin = np.linspace(0.3*self.g.dr, dmax, self.n_bins)

        self.mask = self.radius <= 3*self.dr

    # @jit

    def calcSolHarm(self, l):
        # calculate the solid harmonic function R_lm and I_lm in
        # eq 15 and eq 16

        P_l = self.g.scratch_array()
        self.R_l = self.g.scratch_array()
        self.I_l = self.g.scratch_array()

        for i in range(self.g.nr):
            for j in range(self.g.nz):
                P_l[i, j] = calcLegPolyL(l, self.cosTheta[i, j])

        self.R_l = self.radius**l*P_l
        self.I_l = self.radius**(-l-1)*P_l

    # @jit

    def calcML(self):
        # calculate the outer and inner multipole moment function
        # M_lm^R and M_lm^I in eq 17 and eq 18

        self.m_r = np.zeros((self.n_bins), dtype=np.float64)
        self.m_i = np.zeros((self.n_bins), dtype=np.float64)

        for i in range(self.n_bins):
            imask = self.radius <= self.r_bin[i]
            omask = self.radius > self.r_bin[i]
            self.m_r[i] += np.sum(self.R_l[imask] * self.m[imask])
            self.m_i[i] += np.sum(self.I_l[omask] * self.m[omask])

    # @jit

    def sample_mtilde(self, r):
        # calculate the interpolated multipole moment M_lm^R^tilde
        # and M_lm^I^tilde in eq 19

        # we need to find which be we are in
        mu_m = np.argwhere(self.r_bin <= r)[-1][0]
        mu_p = np.argwhere(self.r_bin > r)[0][0]

        assert mu_p == mu_m + 1

        mtilde_r = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_r[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_r[mu_m]

        mtilde_i = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_i[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_i[mu_m]
        return mtilde_r, mtilde_i

    # @jit
    def calcMulFace(self, dr, dz, l):
        # calculate the contribution of M_lm^R^tilde * conj(I_lm) +
        # conj(M_lm^I^tilde)*R_lm at the surface
        # evaluated at the face of the cell

        # rho and z coordinates of all surfaces of grid cell
        rFace = self.g.r2d + dr
        zFace = self.g.z2d + dz

        radius = np.sqrt((rFace - self.center[0])**2 +
                         (zFace - self.center[1])**2)

        cosTheta = zFace/radius

        mulFace_l = self.g.scratch_array()
        mtilde_r = self.g.scratch_array()
        mtilde_i = self.g.scratch_array()
        R_l = self.g.scratch_array()
        I_l = self.g.scratch_array()

        for i in range(self.g.nr):
            for j in range(self.g.nz):
                '''
                mtilde_r, mtilde_i = self.sample_mtilde(
                    radius[i, j])

                P_l = calcLegPolyL(l, cosTheta[i, j])

                R_l = radius[i, j]**l * P_l
                I_l = radius[i, j]**(-l-1) * P_l

                mulFace_l[i, j] = mtilde_r * I_l + mtilde_i * R_l
                '''
                mtilde_r[i, j], mtilde_i[i, j] = self.sample_mtilde(radius[i, j])

                P_l = calcLegPolyL(l, cosTheta[i, j])

                R_l[i, j] = radius[i, j]**l * P_l
                I_l[i, j] = radius[i, j]**(-l-1) * P_l

        mulFace_l = mtilde_r * I_l + mtilde_i * R_l

        return mulFace_l

    # @ jit
    def Phi(self):

        dr = self.g.dr/2
        dz = self.g.dz/2

        area_m_r = 2*np.pi*(self.g.r2d-dr)*dz
        area_m_z = np.pi*((self.g.r2d+dr)**2 - (self.g.r2d-dr)**2)
        area_p_r = 2*np.pi*(self.g.r2d+dr)*dz
        area_p_z = np.pi*((self.g.r2d+dr)**2 - (self.g.r2d-dr)**2)
        total_area = area_m_r+area_m_z +\
            area_p_r+area_p_z

        phi = self.g.scratch_array()

        # for l in range(self.lmax+1):
        for l in range(0, self.lmax+1, 2):
            # in case where the mass is symetric, the multipole of odd l vanish
            self.calcSolHarm(l)
            self.calcML()
            MulFace_minus_r = self.calcMulFace(-dr, 0, l)
            MulFace_minus_z = self.calcMulFace(0, -dz, l)
            MulFace_plus_r = self.calcMulFace(dr, 0, l)
            MulFace_plus_z = self.calcMulFace(0, dz, l)
            '''
            phi += -sc.G*(MulFace_minus_r +
                          MulFace_minus_z +
                          MulFace_plus_r +
                          MulFace_plus_z)/4
                          '''

            phi += (MulFace_minus_r*area_m_r +
                    MulFace_minus_z*area_m_z +
                    MulFace_plus_r*area_p_r +
                    MulFace_plus_z*area_p_z)

        phi = -sc.G*phi/total_area

        # phi = phi/total_area

        return phi


"""
@jit
def calcR_l(l, z, r):
    # Calculate the solid harmonic function R_l^c for 2d axisymmetric condition
    # We use the recurrence relation defined in Flash user guide
    # R_l^c = ((2*l-1)*z*R_{l-1}^c-r**2*R_{l-2}^c)/l**2
    # This recurrence reltaion use the initial conditions:
    # R_0^c = 1
    # R_1^c = z
    if (l == 0):
        return 1.0
    elif (l == 1):
        return z
    else:
        R_l2 = 1.0
        R_l1 = z
        for n in range(2, l+1):
            # R_l = ((2*n-1)*z*R_l1-r**2*R_l2)/n**2

            # here use another recurrence relation:
            # R_l=((2l-1)*z*R_{l-1}-(l-1)*r^2*R_{l-2})/n
            R_l = ((2*n-1)*z*R_l1-(n-1)*r**2*R_l2)/n
            R_l2 = R_l1
            R_l1 = R_l
        return R_l


@jit
def calcI_l(l, z, r):
    # Calculate the solid harmonic function I_l^c for 2d axisymmetric condition
    # We use the recurrence relation defined in Flash user guide
    # I_l^c = ((2*l-1)*z*I_{l-1}^c-(l-1)**2*R_{l-2}^c)/r**2
    # This recurrence reltaion use the initial conditions:
    # I_0^c = 1/r
    # I_1^c = z/r**3
    if (l == 0):
        return 1/r
    elif(l == 1):
        return z/r**3
    else:
        I_l2 = 1.0
        I_l1 = z/r**3
        for n in range(2, l+1):
            # I_l = ((2*n-1)*z*I_l1-(n-1)**2*I_l2)/r**2

            # here use another recurrence relation:
            # I_l = ((2l-1)*z*I_{l-1}-(l-1)*I_{l-2})/(r^2*l)
            I_l = ((2*n-1)*z*I_l1-(n-1)*I_l2)/(r**2*n)
            I_l2 = I_l1
            I_l1 = I_l
        return I_l


def DampFactorR(self):
    Lfactorial = 1.0
    for l in range(self.lmax+1):
        Lfactorial = Lfactorial*l


class Multipole2d():
    '''The Multipole Expansion in 2d case'''

    def __init__(self, grid, lmax, dr, density, center=(0.0, 0.0)):

        self.g = grid
        self.lmax = lmax
        self.dr = dr
        self.density = density
        self.center = center
        self.r = grid.r2d - self.center[0]
        self.z = grid.z2d - self.center[1]
        self.radius = np.sqrt(self.r**2 + self.z**2)

        '''
        self.radius = np.sqrt((self.g.r2d - self.center[0])**2 +
                              (self.g.z2d - self.center[1])**2)
                              '''

        self.m = self.density * self.g.vol

        self.phi = self.g.scratch_array()

        # compute the bins
        # this computation method is not correct
        r_max = max(abs(self.g.rlim[0] - center[0]), abs(self.g.rlim[1] -
                                                         center[0]))
        z_max = max(abs(self.g.zlim[0] - center[1]), abs(self.g.zlim[1] -
                                                         center[1]))

        # radii of concentric spheres
        dmax = np.sqrt(r_max**2 + z_max**2)

        self.n_bins = int(dmax/dr)

        self.r_bin = np.linspace(0, dmax, self.n_bins)

        self.mask = self.radius <= 3*self.dr

    @jit
    def calcML(self, l):
        # calculate the solid harmonic function R_lm and I_lm in
        # eq 15 and eq 16

        R_l = self.g.scratch_array()
        I_l = self.g.scratch_array()

        for i in range(self.g.nr):
            for j in range(self.g.nz):
                R_l[i, j] = calcR_l(l, self.z[i, j], self.radius[i, j])
                I_l[i, j] = calcI_l(l, self.z[i, j], self.radius[i, j])


        self.m_r = np.zeros((self.n_bins), dtype=np.float64)
        self.m_i = np.zeros((self.n_bins), dtype=np.float64)

        for i in range(self.n_bins):
            imask = self.radius <= self.r_bin[i]
            omask = self.radius > self.r_bin[i]
            self.m_r[i] += np.sum(R_l[imask] * self.m[imask])
            self.m_i[i] += np.sum(I_l[omask] * self.m[omask])

    @jit
    def sample_mtilde(self, r):
        # calculate the interpolated multipole moment M_lm^R^tilde
        # and M_lm^I^tilde in eq 19

        # we need to find which be we are in
        mu_m = np.argwhere(self.r_bin <= r)[-1][0]
        mu_p = np.argwhere(self.r_bin > r)[0][0]

        assert mu_p == mu_m + 1

        mtilde_r = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_r[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_r[mu_m]

        mtilde_i = (r - self.r_bin[mu_m])/(self.r_bin[mu_p] - self.r_bin[mu_m]
                                           ) * self.m_i[mu_p] + \
            (r - self.r_bin[mu_p])/(self.r_bin[mu_m] -
                                    self.r_bin[mu_p]) * self.m_i[mu_m]
        return mtilde_r, mtilde_i

    @jit
    def calcMulFace(self, dr, dz, l):
        # calculate the contribution of M_lm^R^tilde * conj(I_lm) +
        # conj(M_lm^I^tilde)*R_lm at the surface
        # evaluated at the face of the cell

        # rho and z coordinates of all surfaces of grid cell
        rFace = self.r + dr
        zFace = self.z + dz

        '''
        radius = np.sqrt((rFace - self.center[0])**2 +
                         (zFace - self.center[1])**2)
                         '''

        radius = np.sqrt(rFace**2 + zFace**2)

        mulFace_l = self.g.scratch_array()
        mtilde_r = self.g.scratch_array()
        mtilde_i = self.g.scratch_array()
        R_l = self.g.scratch_array()
        I_l = self.g.scratch_array()

        for i in range(self.g.nr):
            for j in range(self.g.nz):
                '''
                mtilde_r, mtilde_i = self.sample_mtilde(
                    radius[i, j])

                P_l = calcLegPolyL(l, cosTheta[i, j])

                R_l = radius[i, j]**l * P_l
                I_l = radius[i, j]**(-l-1) * P_l

                mulFace_l[i, j] = mtilde_r * I_l + mtilde_i * R_l
                '''
                mtilde_r[i, j], mtilde_i[i, j] = self.sample_mtilde(radius[i, j])

                R_l[i, j] = calcR_l(l, zFace[i, j], radius[i, j])
                I_l[i, j] = calcI_l(l, zFace[i, j], radius[i, j])

        mulFace_l = mtilde_r * I_l + mtilde_i * R_l

        return mulFace_l

    @ jit
    def Phi(self):

        dr = self.g.dr/2
        dz = self.g.dz/2

        '''
        area_m_r = 2*np.pi*(self.g.r2d-dr)*dz
        area_m_z = np.pi*((self.g.r2d+dr)**2 - (self.g.r2d-dr)**2)
        area_p_r = 2*np.pi*(self.g.r2d+dr)*dz
        area_p_z = np.pi*((self.g.r2d+dr)**2 - (self.g.r2d-dr)**2)
        total_area = area_m_r+area_m_z +\
            area_p_r+area_p_z
            '''

        phi = self.g.scratch_array()

        # for l in range(self.lmax+1):
        for l in range(0, self.lmax+1, 2):
            # in case where the mass is symetric, the multipole of odd l vanish
            self.calcML(l)
            MulFace_minus_r = self.calcMulFace(-dr, 0, l)
            MulFace_minus_z = self.calcMulFace(0, -dz, l)
            MulFace_plus_r = self.calcMulFace(dr, 0, l)
            MulFace_plus_z = self.calcMulFace(0, dz, l)

            phi += (MulFace_minus_r +
                    MulFace_minus_z +
                    MulFace_plus_r +
                    MulFace_plus_z)/4

            '''
            phi += (MulFace_minus_r*area_m_r +
                    MulFace_minus_z*area_m_z +
                    MulFace_plus_r*area_p_r +
                    MulFace_plus_z*area_p_z)
                    '''

        # phi = -sc.G*phi/total_area

        # phi = phi/total_area

        phi = -sc.G*phi

        return phi
        """
