'''
testing the functions from calc_energies.py
'''
import cvxpy as cp
import numpy as np
import time
from datetime import timedelta
from line_profiler import LineProfiler
from multiprocessing import Pool
import os
from itertools import repeat
import define_SDP_cg as dsdp_cg
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy

import calc_energies as ce
import solve_SDP as ssdp
import define_system as ds


def testing_check_MPS(N, pos=0):
    return ce.check_MPS(N, pos=pos)
'''
testing_check_MPS(5, pos=1)
mps = ce.prepare_MPS_state('heisenberg3', 5)
print(mps)
'''

def testing_cp_var_into_np_var():
    N=4
    rhos = ssdp.rhos_cp_variable(N)
    ham = ds.Build_H_XXZ(Delta=1)
    #ham = ds.Build_H_TFIM_Interaction_Term()
    prob, prob2 = ce.cp_var_into_np_var(ham, rhos, N=N)
    return prob, prob2

#prob, prob2 = testing_cp_var_into_np_var()

def testing_profiling_cp_vs_np_ptrace():
    lp = LineProfiler()
    lp_wrapper = lp(ce.profiling_cp_vs_np_ptrace)
    lp_wrapper(dim=2)
    lp.print_stats()
    return print('finish')
#testing_profiling_cp_vs_np_ptrace()

def testing_profiling_syspermute():
    lp = LineProfiler()
    lp_wrapper = lp(ce.profiling_syspermute)
    lp_wrapper(dim=2)
    lp.print_stats()
    return print('finish')
#testing_profiling_syspermute()


def testing_profiling_cp_to_np_converion():
    lp = LineProfiler()
    lp_wrapper = lp(ce.profiling_cp_to_np_converion)
    lp_wrapper(dim=2)
    lp.print_stats()
    return print('finish')
#testing_profiling_cp_to_np_converion()

def testing_cp_to_np_parallel():
    dim = 2
    chi = 1
    #rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    omega4 = cp.Variable((dim**(12)*chi, dim**(12)*chi), PSD=True)
    cvx_obj = omega4
    data = 0
    t0 = time.time()
    if __name__ == '__main__':
        print('inside pool stuff')
        pool = Pool(os.cpu_count()-1)
        data = pool.starmap(ce.cp_to_np_parallel, zip(repeat(cvx_obj), range(cvx_obj.shape[0])))
        #pool.map(ce.cp_to_np_parallel2, range(cvx_obj.shape[0]))
        #print(data)
    t1 = time.time()
    total_time = t1-t0
    print('total time using parallelized code =', timedelta(seconds=total_time))

    return data
#data = testing_cp_to_np_parallel()

def compare_cp_to_np_non_parallel():
    '''
    compare to non parallelized code
    '''
    dim = 2
    chi = 1
    #rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    '''
    takes super much memory
    '''
    omega4 = cp.Variable((dim**(12)*chi, dim**(12)*chi), PSD=True)
    t0 = time.time()
    cvx_obj = cvxpy_to_numpy(omega4)
    t1 = time.time()
    total_time = t1-t0
    print('total time using non parallelized code =', timedelta(seconds=total_time))


    return print('finish')
#compare_cp_to_np_non_parallel()

'''
testing the partial trace function for tracing out multiple subsystems
'''
def testing_partial_trace_cp():
    dim = 2
    rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    sys = [1,2,3,6,9]
    dim = list(np.ones(9, dtype='int')*dim)
    res = ce.partial_trace_cp(rho3, sys, dim)
    return 0
#testing_partial_trace_cp()


def testing_calc_energies_2D():
    ham = ds.Build_H_XXZ(Delta=1)
    #ham = ds.Build_H_TFIM_Interaction_Term()
    W = dsdp_cg.generate_linear_map_2D(dim=2, chi=1)
    t0 = time.time()
    prob, constraints = ce.calc_energies_2D(ham, W, dim=2, chi=1)
    t1 = time.time()
    total_time = t1-t0
    print('total time =', timedelta(seconds=total_time))
    return prob, constraints

prob, constraints = testing_calc_energies_2D()





































#
