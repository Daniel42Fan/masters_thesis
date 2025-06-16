"""
just a playground to test some stuff
"""

import numpy as np
from numpy import linalg as nLA
from scipy import linalg as sLA
import random
import scipy.sparse as sparse
import cvxpy as cp
import matplotlib.pyplot as plt
import pickle as pickle
import os
import tensornetwork as tn
import itertools as it
import qutipy.general_functions as qt_gf
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy

import define_SDP as dsdp
import define_SDP_cg as dsdp_cg
import define_system as ds
import solve_SDP as ssdp
import solve_SDP_cg as ssdp_cg
import utility as ut
import calc_energies as ce
import analyze_data as ad


"""
def add(a, b ,c):
    return a + b +c

def subtract(a, b ,c):
    return a - b - c

def calc(func):
    res = func(1,2,3)
    return res

print(calc(add))
print(calc(subtract))
"""
N = 5
chi = 1
dim = 2
Delta = 1
H = ds.Build_H_XXZ_full(N - 2, dim=dim, Delta=Delta)
# print('H\n', H)
"""
eigvals, eigvecs = ut.eigen(H, chi)
eigvals2, eigvecs2 = sLA.eigh(H)


print('eigvals =', eigvals)
print('eigvals.shape =', eigvals.shape)
print('eigvecs =', eigvecs)
print('eigvecs.shape =', eigvecs.shape)

print('eigvals2 =', eigvals2)
print('eigvals2.shape =', eigvals2.shape)
print('eigvecs2 =\n', eigvecs2)
print('eigvecs2.shape =', eigvecs2.shape)



W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=0, a=-1, b=1)
print('W =', W)
print('W.shape =', W.shape)
W0 = eigvecs.T + 0.1*W
print('W0', W0)

"""
"""
Delta = 1
N = 4
chi = 1
dim = 2
seed = 0
W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed)
print('W=\n', W)
print('W.shape =', W.shape)
#W = ad.get_optimized_cg_map(Delta, N, chi)
H = ds.Build_H_XXZ_full(N-2, Delta=Delta)
print('Hamiltonian XXZ for N={}, chi={}\n'.format(N, chi), H)
# store the Qs into a list,a fter QR decomposition



ham_seeds = [3,6,9]
#for seed_ham in range(2, 3):
for seed_ham in range(1,11):
    print('Random Hamiltonian seed {}'.format(seed_ham))
    N = 5
    ham = ds.Build_random_H(seed_ham)
    prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(N-1, N, ham, dim=2, output=False)
    print('energy for {} ='.format(N-1), prob_N_minus_1.value)
    print('energy for {} ='.format(N), prob_N.value)
    print('energy difference =', prob_N_minus_1.value-prob_N.value)
    print('############################################################\n')
"""
"""
N = 5
Delta = 1
print(ds.Build_H_XXZ_full(N-2, Delta=Delta).shape)
ham_rand = ds.Build_random_H(3)
ham_rand_full = ds.Build_H_full(N-2, ham_rand)
print(ham_rand.shape)
print(ham_rand_full.shape)
"""

a = [4, 7, 2, 9, 1]
print(not a)


#
