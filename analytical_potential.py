import numpy as np
import scipy.constants as sc


# class of various kinds of object
class Ana_Mac_pot(object):
    """Analytical potential of MacLaurin spheroid"""

    def __init__(self, a_1, a_3, rho, z, density):
        assert a_1 > a_3
        e = np.sqrt(1 - (a_3/a_1)**2)

        # potential inside the sphere
        self.potential = 0.0

        if z**2/a_3**2+rho**2/a_1**2 <= 1:
            A1 = np.sqrt(1 - e**2)/e**3*np.arcsin(e)-(1-e**2)/e**2
            A3 = 2/e**2-2*np.sqrt(1 - e**2)/e**3*np.arcsin(e)

            self.potential = -np.pi*sc.G*density *\
                (2*A1*a_1**2-A1*rho**2 +
                 A3*(a_3**2-z**2))

        else:
            b = a_1**2+a_3**2-rho**2-z**2
            c = a_1**2*a_3**2-a_3**2 * rho**2 - z**2*a_1**2
            # lambda is the positive root of the eq. 27 of arXiv:1307.3135
            # lambda = (-b+sqrt(b**2+4ac))/2a
            lam = (-b+np.sqrt(b**2-4*c))/2
            assert lam >= 0
            h = a_1*e/np.sqrt(a_3**2+lam)

            self.potential = -2*a_3/e**2*np.pi*sc.G * density *\
                (a_1*e*np.arctan(h)-1/2 * rho**2 *
                 (np.arctan(h)-h/(1+h**2)) + 2*z**2 *
                 (h-np.arctan(h)))


class Ana_Sph_pot(object):
    """Analytical potential of a perfect sphere"""

    def __init__(self, rho, z, a, density):
        r = np.sqrt(z**2+rho**2)

        self.potential = 0.0

        if r <= a:
            self.potential = -2/3*np.pi*sc.G*density*(3*a**2-r**2)

        else:
            self.potential = -4/3*np.pi*sc.G*density*(a**3/r)
