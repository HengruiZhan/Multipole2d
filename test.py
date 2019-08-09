import analytical_potential as ap
import comparitive_difference as comdf
import numpy as np
import matplotlib.pyplot as plt
import grid
import L2_difference as L2
import multipole


nr = 128
nz = 255


rlim = (0, 0.1)
zlim = (-1.0, 1.0)
g = grid.Grid(nr, nz)

dens = g.scratch_array()

"""
print(g.r.shape)
print(g.z.shape)
print(np.amax(g.r))
print(np.amax(g.z))
print(np.amin(g.r))
print(np.amin(g.z))
"""

# density of a perfect sphere
sph_center = (0.0, 0.0)
radius = np.sqrt((g.r - sph_center[0])**2 + (g.z - sph_center[1])**2)
a = 0.3
dens[radius <= a] = 1.0

density = 1.0
"""
phi_sph = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        sph_phi = ap.Ana_Sph_pot(g.r[i], g.z[j], a, density)
        phi_sph[i, j] = sph_phi.potential
"""
"""
# density of a MacLaurin spheroid
sph_center = (0.0, 0.0)
a_1 = 0.23
a_3 = 0.10
mask = g.r2d**2/a_1**2 + g.z2d**2/a_3**2 <= 1
dens[mask] = 1.0
density = 1.0

phi_mac = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        mac_phi = ap.Ana_Mac_pot(a_1, a_3, g.r[i], g.z[j], density)
        phi_mac[i, j] = mac_phi.potential
"""
"""
plt.imshow(np.log10(np.abs(np.transpose(phi_mac))), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])

plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("anal_phi.png")
"""
"""
plt.imshow(np.transpose(dens), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("dens.png")
e = np.sqrt(1 - (a_3/a_1)**2)
"""

center = (0.0, 0.0)
n_moments = 30
m = multipole.Multipole(g, n_moments, 2*g.dr, center=center)


m.compute_expansion(dens)
phi = g.scratch_array()

for i in range(g.nr):
    for j in range(g.nz):
        phi[i, j] = m.phi(g.r[i, j], g.z[i, j])

# phi = m.phi(g.r, g.z)

plt.imshow(np.log10(np.abs(np.transpose(phi))), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("lmax=1.png")
"""
L2normerr = L2.L2_diff(phi_sph, phi, g.scratch_array())
print(L2normerr)
"""
"""
xm = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ym = np.array([10, 9.5, 7.2, 4.0, 3.5, 3.3, 3.2, 3.1, 2.9, 2.7, 2.5])
"""
