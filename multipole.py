import numpy as np
import scipy.constants as sc
from scipy.special import sph_harm


"""
class Multipole():
    '''The Multipole is written in vetorized codes'''

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

    def compute_sph_harm(self, l, r, z):
        # r and z are all array
        # tan(theta) = r/z
        theta = np.arctan2(r, z)

        Y_lm = sph_harm(0, l, 0.0, theta)

        return Y_lm

    def compute_harmonics(self, l, r, z):
        # r and z are all array
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
            '''
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
            '''
            # harmonic function at r-dr/2 surface
            Y_lm_r_minus = self.compute_sph_harm(l, r-self.g.dr/2, z)
            # harmonic function at r+dr/2 surface
            Y_lm_r_plus = self.compute_sph_harm(l, r+self.g.dr/2, z)
            # harmonic function at r-dz/2 surface
            Y_lm_z_minus = self.compute_sph_harm(l, r, z-self.g.dz/2)
            # harmonic function at r+dz/2 surface
            Y_lm_z_plus = self.compute_sph_harm(l, r, z+self.g.dz/2)
            # average harmonic function of all the surface
            Y_lm = 1/4*(Y_lm_r_minus+Y_lm_r_plus+Y_lm_z_minus+Y_lm_z_plus)
            R_lm = np.sqrt(4*np.pi/(2*l + 1)) * radius**l * Y_lm
            I_lm = np.nan_to_num(np.sqrt(4*np.pi/(2*l + 1)) *
                                 Y_lm / radius**(l+1))
            # calculate Eq. 20
            phi_zone += sc.G * (mtilde_r * np.conj(I_lm) +
                                np.conj(mtilde_i) * R_lm)

        return -np.real(phi_zone)
    """


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

# bug


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
        return R_l

# bug


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
            # I_l = ((2l-1)*z*I_{l-1}-(l-1)*I_{l-2})/(R^2*l)
            I_l = ((2*n-1)*z*I_l1-(n-1)*I_l2)/(r**2*n)
        return I_l


"""
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

    '''
    def DampFactorR(self):
        Lfactorial = 1.0
        for l in range(self.lmax+1):
            Lfactorial = Lfactorial*l
        '''

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

        # debug codes
        print("l = ", l)
        if (l == self.lmax):
            print("cell center R_", l, "near the center:", self.R_l[self.mask])
            print("the miniimun of R_", l, ":",
                  np.amin(np.absolute(self.R_l)))
            print("cell center I_", l, "near the center:", self.I_l[self.mask])
            print("the maximun of I_", l, ":",
                  np.amax(np.absolute(self.I_l)))

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

        '''
        # debug codes
        print("m_r = ", self.m_r)
        print("m_i = ", self.m_i)
        '''

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

        '''
        # degug codes
        print("l=", l)
        print("dr=", dr)
        print("dz=", dz)
        # print out every terms near the expansion center
        mask = self.radius <= self.dr
        print("mtilde_r near the center are", mtilde_r[mask])
        print("I_l near the center are", I_l[mask])
        print("mtilde_i near the center are", mtilde_i[mask])
        print("R_l near the center are", R_l[mask])
        '''

        return mulFace_l

    # @ jit
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
            self.calcSolHarm(l)
            self.calcML()
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

        phi = -sc.G*phi/total_area
        '''
        # phi = phi/total_area

        '''
        # degug codes
        max = np.amax(np.abs(phi))
        min = np.amin(np.abs(phi))
        print("maximun of potential is", max)
        mask_max = np.abs(phi) == max
        print("maximun of potential is at", np.argwhere(mask_max))
        print("minimun of potential is", min)
        mask_min = np.abs(phi) == min
        print("maximun of potential is at", np.argwhere(mask_min))
        '''
        '''
        mask = self.radius <= 3*self.dr
        print("potential near the center:", np.abs(phi[mask]))
        '''

        return phi
        """


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

        '''
        # debug codes
        print("l = ", l)
        if (l == self.lmax):
            print("cell center R_", l, "near the center:", self.R_l[self.mask])
            print("the miniimun of R_", l, ":",
                  np.amin(np.absolute(self.R_l)))
            print("cell center I_", l, "near the center:", self.I_l[self.mask])
            print("the maximun of I_", l, ":",
                  np.amax(np.absolute(self.I_l)))
                  '''

        self.m_r = np.zeros((self.n_bins), dtype=np.float64)
        self.m_i = np.zeros((self.n_bins), dtype=np.float64)

        for i in range(self.n_bins):
            imask = self.radius <= self.r_bin[i]
            omask = self.radius > self.r_bin[i]
            self.m_r[i] += np.sum(R_l[imask] * self.m[imask])
            self.m_i[i] += np.sum(I_l[omask] * self.m[omask])

        '''
        # debug codes
        print("m_r = ", self.m_r)
        print("m_i = ", self.m_i)
        '''

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

        '''
        # degug codes
        print("l=", l)
        print("dr=", dr)
        print("dz=", dz)
        # print out every terms near the expansion center
        mask = self.radius <= self.dr
        print("mtilde_r near the center are", mtilde_r[mask])
        print("I_l near the center are", I_l[mask])
        print("mtilde_i near the center are", mtilde_i[mask])
        print("R_l near the center are", R_l[mask])
        '''

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

        '''
        # degug codes
        max = np.amax(np.abs(phi))
        min = np.amin(np.abs(phi))
        print("maximun of potential is", max)
        mask_max = np.abs(phi) == max
        print("maximun of potential is at", np.argwhere(mask_max))
        print("minimun of potential is", min)
        mask_min = np.abs(phi) == min
        print("maximun of potential is at", np.argwhere(mask_min))
        '''
        '''
        mask = self.radius <= 3*self.dr
        print("potential near the center:", np.abs(phi[mask]))
        '''

        return phi
