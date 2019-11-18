import numpy as np


def L2_diff(analytical_phi, approximate_phi, grid):
    phi_difference = analytical_phi - approximate_phi
    '''
    N = grid.scratch_array().size
    L2_norm = np.sqrt(np.sum(np.square(phi_difference)/N))
    analytical_norm = np.sqrt(np.sum(analytical_phi**2/N))
    '''
    L2_norm = np.sqrt(np.sum(np.square(phi_difference)))
    analytical_norm = np.sqrt(np.sum(analytical_phi**2))
    L2_norm = L2_norm/analytical_norm
    return L2_norm


'''
def L2_diff(analytical_phi, approximate_phi, grid):
    phi_difference = analytical_phi - approximate_phi
    L2_norm = np.sqrt(np.sum(phi_difference**2*grid.vol_project))
    analytical_norm = np.sqrt(np.sum(analytical_phi**2*grid.vol_project))
    L2_norm = L2_norm/analytical_norm

    return L2_norm
    '''

'''
def L2_diff(analytical_phi, approximate_phi, grid):
    phi_difference = analytical_phi - approximate_phi
    L2_norm = np.sqrt(np.sum(phi_difference**2*grid.vol))
    analytical_norm = np.sqrt(np.sum(analytical_phi**2*grid.vol))
    L2_norm = L2_norm/analytical_norm

    return L2_norm
    '''
