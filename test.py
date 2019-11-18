import analytical_potential as ap
import comparitive_difference as comdf
import numpy as np
import matplotlib.pyplot as plt
import grid
import L1_difference as L1
import L2_difference as L2
import multipole
import multipole_bruteforce


nr = 8
nz = 16

nr = 16
nz = 32


nr = 32
nz = 64


nr = 64
nz = 128
'''
nr = 128
nz = 256
'''

rlim = (0, 0.5)
zlim = (-0.5, 0.5)
'''
rlim = (0, 0.5)
zlim = (-0.5, 0.5)
'''
'''
rlim = (0, 1.0)
zlim = (-1.0, 1.0)
'''
# g = grid.Grid(nr, nz)
g = grid.Grid(nr, nz, rlim, zlim)

'''
rlim = (0, 1000)
zlim = (-1000, 1000)
g = grid.Grid(nr, nz, rlim, zlim)
'''
'''
# test very large coordinates
rlim = (0, 1.0*10**10)
zlim = (-1.0*10**10, 1.0*10**10)
g = grid.Grid(nr, nz, rlim, zlim)
'''

dens = g.scratch_array()
'''
# density of a perfect sphere
sph_center = (0.0, 0.0)
radius = np.sqrt((g.r2d - sph_center[0])**2 + (g.z2d - sph_center[1])**2)
# a = 0.5
# a = 0.25
# a = g.dr
a = 0.25
dens[radius <= a] = 1.0
density = 1.0

phi_anal = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        sph_phi = ap.Ana_Sph_pot(g.r2d[i, j], g.z2d[i, j], a, density)
        phi_anal[i, j] = sph_phi.potential
        '''

'''
# normalized density of a sphere
m = 4/3*np.pi*a**3*density
# actural volume of a sphere
mask_mass = (g.r2d)**2 + (g.z2d)**2 <= a**2
vol_mass = np.sum(g.vol[mask_mass])

density_norm = m/vol_mass

dens_norm = g.scratch_array()
dens_norm[mask_mass] = density_norm
'''
'''
# density of double sphere
a = 0.1
density = 1.0
sph1_center = (0.0, 0.5)
sph2_center = (0.0, -0.5)
mask1 = (g.r2d-sph1_center[0])**2 + (g.z2d-sph1_center[1])**2 <= a**2
dens[mask1] = 1.0
mask2 = (g.r2d-sph2_center[0])**2 + (g.z2d-sph2_center[1])**2 <= a**2
dens[mask2] = 1.0

# normalized density of double spheres
m = 4/3*np.pi*a**3*density
# actural volume of a sphere
mask_mass = (g.r2d)**2 + (g.z2d)**2 <= a**2
vol_mass = np.sum(g.vol[mask_mass])

density_norm = m/vol_mass

dens_norm = g.scratch_array()
dens_norm[mask1] = density_norm
dens_norm[mask2] = density_norm

# analytical potential of double sphere
phi_anal = g.scratch_array()
for i in range(g.nr):
    for j in range(g.nz):
        double_sph = ap.Ana_Double_Sph_pot(g.r[i], g.z[j], sph1_center,
                                           sph2_center, a, density)
        phi_anal[i, j] = double_sph.potential
        '''

# density of a MacLaurin spheroid
sph_center = (0.0, 0.0)

a_3 = 0.10
e = 0.9
a_1 = a_3/np.sqrt(1-e**2)

mask = g.r2d**2/a_1**2 + g.z2d**2/a_3**2 <= 1
dens[mask] = 1.0
density = 1.0

# normalized density of a spheroid
m = 4/3*np.pi*a_1**2*a_3*density
# actural volume of a sphere
mask_mass = mask
vol_mass = np.sum(g.vol[mask_mass])
density_norm = m/vol_mass
dens_norm = g.scratch_array()
dens_norm[mask_mass] = density_norm

# analytical solution for the MacLaurin spheroid
phi_anal = g.scratch_array()
for i in range(nr):
    for j in range(nz):
        mac_phi = ap.Ana_Mac_pot(a_1, a_3, g.r2d[i, j], g.z2d[i, j], density)
        phi_anal[i, j] = mac_phi.potential
'''
plt.imshow(np.log10(np.abs(np.transpose(phi_anal))), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("spheroid_a3=0.1_e=0.9.png")
'''

center = (0.0, 0.0)
lmax = 30
m = multipole.Multipole2d(g, lmax, 0.3*g.dr, dens_norm, center=(0.0, 0.0))
# m = multipole.Multipole2d(g, lmax, 0.3*g.dr, dens, center=(0.0, 0.0))
# test bruteforce algorithm
# m = multipole_bruteforce.Multipole2d(g, lmax, dens_norm, center=(0.0, 0.0))

# phi = g.scratch_array()

# phi = m.Phi(512, 1024)
phi = m.Phi()

plt.imshow(np.log10(np.abs(np.transpose(phi))), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("lmax=30.png")


diff = comdf.comp_diff(phi_anal, phi)
# print(np.amin(np.abs(diff.difference)))
plt.imshow(np.abs(np.transpose(diff.difference)), origin="lower",
           interpolation="nearest",
           extent=[g.rlim[0], g.rlim[1],
                   g.zlim[0], g.zlim[1]])
plt.colorbar()
ax = plt.gca()
ax.set_aspect("equal")
plt.savefig("comparitive_difference0_32.png")

print("resolution is", nr)
print("lmax =", lmax)

L2normerr = L2.L2_diff(phi_anal, phi, g)
print("L2 norm error is", L2normerr)

'''
L1normerr = L1.L1_diff(phi_anal, phi, g.scratch_array())
print("L1 norm error is", L1normerr)
'''
