import numpy as np
import cvxpy as cp
import scipy.sparse as sparse
import time
from datetime import timedelta
import qutipy.general_functions as qt_gf
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy
import calc_energies as ce

import utility as ut
import define_SDP as dsdp
import define_SDP_cg as dsdp_cg
import define_system as ds



'''
create positive semidefinite matrix as optimization variable
for our case, all the rhos are the variables
'''
def rhos_cp_variable(N, dim=2):
    rhos = []
    d = dim
    #include the N-th index
    for i in range(2, N+1):
        rhos.append(cp.Variable((d**i,d**i), PSD=True))
    return rhos

'''
create a single rho as a cp variable
'''
def get_rho_cp_variable(N, dim=2):
    rho = cp.Variable((dim**N,dim**N), PSD=True)
    return rho

'''
sdp as human readable problem
'''
def Minimize_trace_rho_times_h(ham, rhos, N=4, output=True):
    # system size
    # N
    rho2 = rhos[0]
    # create constraints
    constraints = []

    constraints.append(cp.trace(rho2) == 1)
    for i in range(0, len(rhos)-1):
        rho_shape_0 = rhos[i+1].shape[0]
        #print('timing using cp ptr')
        #t0 = time.time()
        ptr_R = cp.partial_trace(rhos[i+1], [int(rho_shape_0/2), 2], axis=1)
        ptr_L = cp.partial_trace(rhos[i+1], [2, int(rho_shape_0/2)], axis=0)
        #t1 = time.time()
        #total_time = t1-t0
        #print('total time using cp ptr =', timedelta(seconds=total_time))
        constraints.append(ptr_R == rhos[i])
        constraints.append(ptr_L == rhos[i])
    #print(len(constraints))
    # form objective
    #print('rho2.shape =\n', rho2.shape)
    obj = cp.Minimize(cp.trace(rho2@ham))

    # form and solve problem
    prob = cp.Problem(obj, constraints)
    #prob.solve(solver=cp.SCS)
    #prob.solve(verbose=True)
    prob.solve()
    if output==True:
        print("\nHuman readable SDP NO cg")
        print('N =', N)
        print("\nRESULTS:\n")
        print("status:", prob.status)
        print("optimal value", prob.value)
        #print(cp.trace((rho2@ham)).value)
        #print(rhos[0].value)
        #print(rhos[0].value)
        #print(rhos[1].value)
        #print(rhos[2].value)
        print("-------------------------------------------------------")
    else:
        pass
    return prob#, rhos

'''
to compare solutions from a calculatio with cg, the result of the
problem size N with cg must lie between N and N-1 with no cg.
This function returns the non cg solutions
'''
def compare_solutions_no_cg(N_minus_1, N, ham, dim=2, output=False):
    rhos_N_minus_1 = rhos_cp_variable(N_minus_1, dim=dim)
    rhos_N = rhos_cp_variable(N, dim=dim)
    prob_N_minus_1 = Minimize_trace_rho_times_h(ham, rhos_N_minus_1, N=N_minus_1, output=output)
    prob_N = Minimize_trace_rho_times_h(ham, rhos_N, N=N, output=output)
    return prob_N_minus_1, prob_N
'''
compare the energies of size N and M = N-1 by normalizing them to 0 and 1
the energies with cg before and after the grad descent should also be displayed
between 0 and 1.
'''
def compare_normalized_solutions(N_minus_1, N, energy_no_gd, energy_gd, ham, dim=2, output=True):
#def compare_normalized_solutions(N_minus_1, N, ham, dim=2):
    '''
    MAYBE CHANGE ORDER OF THE ENERGIES
    '''
    energies = []
    prob1, prob2 = compare_solutions_no_cg(N_minus_1, N, ham, dim=dim, output=False)
    sol1 = prob1.value
    energies.append(sol1)
    sol2 = prob2.value
    energies.append(sol2)
    #print('sol1 =', sol1)
    #print('sol2 =', sol2)
    # get solution BEFORE grad descent is applied
    energies.append(energy_no_gd)
    # get solution, AFTER grad descent is applied
    energies.append(energy_gd)
    '''
    it is: sol1 < sol2
    and the solution of the cg problem is
    sol1 < sol_cg < sol2
    '''
    # shift the solutions with lower bound to 0:
    shift = sol1
    energies = [e-shift for e in energies]

    sol1_shift = sol1 - shift
    sol2_shift = sol2 - shift
    #print('sol1_shift =', sol1_shift)
    #print('sol2_shift =', sol2_shift)

    # normalize the solutions to 1
    # the largest value (here the shifted sol2) acts as (the inverted)
    # normalization factor
    norm_factor = 1/sol2_shift
    energies = [e*norm_factor for e in energies]
    '''
    sol1_normalized = sol1_shift * norm_factor
    sol2_normalized = sol2_shift * norm_factor
    print('sol1_normalized =', sol1_normalized)
    print('sol2_normalized =', sol2_normalized)
    '''
    if output==True:
        print('normalized energies:')
        print(f'energy no cg for N={N_minus_1}', energies[0])
        print(f'energy w cg for N={N} before gradient descent', energies[2])
        print(f'energy w cg for N={N} after gradient descent', energies[3])
        print(f'energy no cg for N={N}', energies[1])
    else:
        pass
    return energies, shift, norm_factor
    #return sol1_normalized, sol2_normalized

'''
gets the history of intermediate results from gradient descent and normalize
the result
'''
def normalize_history(N_minus_1, N, prob_value, history, ham, dim=2, output=False):
    #history_normalized = []
    shift = 0
    norm_factor = 0
    '''
    for h in history:
        e, shift, norm_factor = compare_normalized_solutions(N_minus_1, N, prob_value, -h, ham, dim=2, output=False)
        history_normalized.append(e[3])
    '''
    e, shift, norm_factor = compare_normalized_solutions(N_minus_1, N, prob_value, -history[0], ham, dim=2, output=False)
    history_normalized = [(-1*h-shift)*norm_factor for h in history]
    return history_normalized

# when considering the old version, where one would needed rho2, rho3...
#def Minimize_trace_rho_times_h_cg(ham, rhos, W, N=4, dim=2, chi=3):
def Minimize_trace_rho_times_h_cg(ham, W, N=4, dim=2, chi=3, output=True):
    # system size
    # N
    dim=dim
    chi=chi

    #print('rhos ', rhos)
    #rho2 = rhos[0]
    '''
    THIS IS HARD CODED
    '''
    #rho3 = rhos[1]
    #rho_N_minus_1 = rhos[N-3]
    rho_N_minus_1 = get_rho_cp_variable(N-1, dim=dim)

    # create constraints
    constraints = []

    '''
    this was in terms of rho2 and rho3
    constraints of ptr rho3 = rho2
    '''
    #constraints.append(cp.trace(rho2) == 1)
    '''
    rho_shape_0 = rhos[1].shape[0]
    ptr_R = cp.partial_trace(rhos[1], [int(rho_shape_0/2), 2], axis=1)
    ptr_L = cp.partial_trace(rhos[1], [2, int(rho_shape_0/2)], axis=0)
    constraints.append(ptr_R == rhos[0])
    constraints.append(ptr_L == rhos[0])
    '''

    # in terms of rho_N_mins_1 and omega_N
    constraints.append(cp.trace(rho_N_minus_1) == 1)

    rho_shape_0 = rho_N_minus_1.shape[0]
    ptr_R = cp.partial_trace(rho_N_minus_1, [int(rho_shape_0/2), 2], axis=1)
    ptr_L = cp.partial_trace(rho_N_minus_1, [2, int(rho_shape_0/2)], axis=0)
    # LTI constraint
    constraints.append(ptr_L == ptr_R)
    # apply coarse graining, in terms of rho2 and rho3
    '''
    rho_cg_L, rho_cg_R, omega = dsdp_cg.coarse_grain_map(rho3, W, dim=dim,
                                                        chi=chi)
    '''
    '''
    get the constraints
    ptr omega_N = V * rho_(N-1) * V_dagger
    '''
    #for i in range(1, N-2):
    #print('begin loop i = ', i)
    # get the the new rho_N, on which a cg map is applied

    # apply coarse graining
    rho_cg_L, rho_cg_R = dsdp_cg.coarse_grain_map_N(rho_N_minus_1, W, N=N,
                                                        dim=dim, chi=chi)

    '''
    rn these rhos are still cp variables
    print('norm of rho_cg_L =\n', np.linalg.norm(rho_cg_L))
    print('norm of rho_cg_R =\n', np.linalg.norm(rho_cg_R))
    '''
    # omega as optimization variable here has shape chi * d^2 x chi * d^2
    omega = cp.Variable((chi*dim**2,chi*dim**2), PSD=True)


    ptr_R = cp.partial_trace(omega, (chi*dim, dim), 1)
    ptr_L = cp.partial_trace(omega, (dim, chi*dim), 0)
    #print('ptr_R.shape', ptr_R.shape)

    constraints.append(ptr_R == rho_cg_R)
    constraints.append(ptr_L == rho_cg_L)
    #print('end loop')

    #print('constraints ', constraints)

    '''
    for i in range(0, len(rhos)-1):
        rho_shape_0 = rhos[i+1].shape[0]
        ptr_R = cp.partial_trace(rhos[i+1], [int(rho_shape_0/2), 2], axis=1)
        ptr_L = cp.partial_trace(rhos[i+1], [2, int(rho_shape_0/2)], axis=0)
        constraints.append(ptr_R == rhos[i])
        constraints.append(ptr_L == rhos[i])
    '''
    # form objective
    #obj = cp.Minimize(cp.trace(rho2@ham))
    obj = cp.Minimize(cp.trace(rho_N_minus_1 @ sparse.kron(ham, sparse.identity(dim**(N-3)))))


    # form and solve problem
    prob = cp.Problem(obj, constraints)
    #prob.solve(solver=cp.MOSEK, verbose=True)
    prob.solve(solver=cp.MOSEK)
    #prob.solve(solver=cp.SCS, verbose=True)
    #prob.solve(verbose=True)
    if output==True:
        print("\nHuman readable SDP with cg")
        print("\nParamters:")
        print("\nN =", N)
        print("dim =", dim)
        print("chi =", chi)
        print("\nRESULTS:\n")
        print("status:", prob.status)
        print("optimal value", prob.value)
        print("for cg map\n", W)

        #print("optimal var", ptr_R.value)
        #print(cp.trace((rho2@ham)).value)
        #print(rhos[0].value)
        '''
        print('rhos[0].value\n', rhos[0].value)
        print('\nrhos[1].value\n', rhos[1].value)
        print('\nrhos[2].value\n', rhos[2].value)
        print('\nomega.value\n', omega.value)
        print('\n rho_cg_L.value\n', rho_cg_L.value)
        '''
        print("-------------------------------------------------------")
    '''
    # overwrite rho3 with the actual value
    rho3 = rhos[1].value
    return prob, rho3, constraints
    '''
    # overwrite rho_N with the actual value
    rho_N_minus_1 = rho_N_minus_1.value
    return prob, rho_N_minus_1, constraints



def Minimize_trace_rho_times_h_cg_2D(ham, W, dim=2, chi=1):
    # system size
    # N
    dim=dim
    chi=chi

    '''
    initialize cg map
    '''
    V = sparse.kron(sparse.identity(dim**5), W)
    '''
    2D rhos, just like in the code of Ilya
    '''
    print('initialize rho and omega variables')
    rho0 = cp.Variable((dim*2, dim*2), PSD=True)
    rho2 = cp.Variable((dim**(2*2), dim**(2*2)), PSD=True)
    rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    omega4 = cp.Variable((dim**(12)*chi, dim**(12)*chi), PSD=True)
    print('finish initialize rho and omega variables')
    '''
    get groundstsate of the 2x2 hamiltonian
    '''
    #V = 0 # TO DO
    #IV = np.kron()
    '''
    calc the 4 partial traces of rho2
    use the qutipy partial trace function
    '''
    print('initialize ptr of rho2')
    '''
    ptr_34 = qt_gf.partial_trace(rho2, [3,4], list(np.ones(4, dtype='int')*dim))
    ptr_12 = qt_gf.partial_trace(rho2, [1,2], list(np.ones(4, dtype='int')*dim))
    ptr_13 = qt_gf.partial_trace(rho2, [1,3], list(np.ones(4, dtype='int')*dim))
    ptr_24 = qt_gf.partial_trace(rho2, [2,4], list(np.ones(4, dtype='int')*dim))
    '''
    ptr_34 = ce.partial_trace_cp(rho2, [3,4], list(np.ones(4, dtype='int')*dim))
    ptr_12 = ce.partial_trace_cp(rho2, [1,2], list(np.ones(4, dtype='int')*dim))
    ptr_13 = ce.partial_trace_cp(rho2, [1,3], list(np.ones(4, dtype='int')*dim))
    ptr_24 = ce.partial_trace_cp(rho2, [2,4], list(np.ones(4, dtype='int')*dim))
    print('finish initialize ptr of rho2')
    '''
    calc the 4 partial traces of rho3

    1 2 3
    4 5 6
    7 8 9

    NE = tracing out system 1,2,3,6,9
    NW = ...
    ...
    '''
    print('initialize ptr of rho3')
    '''
    ptr_rho3_NE = qt_gf.partial_trace(rho3, [1,2,3,6,9], list(np.ones(9, dtype='int')*dim))
    ptr_rho3_NW = qt_gf.partial_trace(rho3, [3,2,1,4,7], list(np.ones(9, dtype='int')*dim))
    ptr_rho3_SE = qt_gf.partial_trace(rho3, [3,6,9,8,7], list(np.ones(9, dtype='int')*dim))
    ptr_rho3_SW = qt_gf.partial_trace(rho3, [1,4,7,8,9], list(np.ones(9, dtype='int')*dim))
    '''
    ptr_rho3_NE = ce.partial_trace_cp(rho3, [1,2,3,6,9], list(np.ones(9, dtype='int')*dim))
    ptr_rho3_NW = ce.partial_trace_cp(rho3, [3,2,1,4,7], list(np.ones(9, dtype='int')*dim))
    ptr_rho3_SE = ce.partial_trace_cp(rho3, [3,6,9,8,7], list(np.ones(9, dtype='int')*dim))
    ptr_rho3_SW = ce.partial_trace_cp(rho3, [1,4,7,8,9], list(np.ones(9, dtype='int')*dim))
    print('finish initialize ptr of rho3')
    '''
    calc partial trace of omega4
    order in omega4
    1 2  3  4
    5 x  x  6
    7 x  x  8
    9 10 11 12  +site 13 (D-dimensional)
    '''
    print('initialize ptr of omega4')
    dims_omega4 = list(np.ones(12, dtype='int')*dim) + [chi]
    '''
    ptr_omega4_NE = qt_gf.partial_trace(omega4, [1,2,3,4,6,8,12], dims_omega4)
    ptr_omega4_NW = qt_gf.partial_trace(omega4, [1,2,3,4,5,7,9], dims_omega4)
    ptr_omega4_SE = qt_gf.partial_trace(omega4, [4,6,8,9,10,11,12], dims_omega4)
    ptr_omega4_SW = qt_gf.partial_trace(omega4, [1,5,7,9,10,11,12], dims_omega4)
    '''
    ptr_omega4_NE = ce.partial_trace_cp(omega4, [1,2,3,4,6,8,12], list(np.ones(12, dtype='int')*dim) + [chi])
    ptr_omega4_NW = ce.partial_trace_cp(omega4, [1,2,3,4,5,7,9], list(np.ones(12, dtype='int')*dim) + [chi])
    ptr_omega4_SE = ce.partial_trace_cp(omega4, [4,6,8,9,10,11,12], list(np.ones(12, dtype='int')*dim) + [chi])
    ptr_omega4_SW = ce.partial_trace_cp(omega4, [1,5,7,9,10,11,12], list(np.ones(12, dtype='int')*dim) + [chi])
    print('finish initialize ptr of omega4')
    '''
    ptr of omega4 should be equal to a permuted system of rho3
    '''
    print('initialize syspermute of rho3')
    # convert cp into np
    rho3 = cvxpy_to_numpy(rho3)
    perm_rho3_NE = qt_gf.syspermute(rho3, [1,4,7,8,9,2,3,5,6], list(np.ones(9, dtype='int')*dim))
    perm_rho3_NW = qt_gf.syspermute(rho3, [3,6,7,8,9,1,2,4,5], list(np.ones(9, dtype='int')*dim))
    perm_rho3_SE = qt_gf.syspermute(rho3, [1,2,3,4,7,5,6,8,9], list(np.ones(9, dtype='int')*dim))
    perm_rho3_SW = qt_gf.syspermute(rho3, [1,2,3,6,9,4,5,7,8], list(np.ones(9, dtype='int')*dim))
    # convert np back to cp
    perm_rho3_list = []
    perm_rho3_NE = numpy_to_cvxpy(perm_rho3_NE)
    perm_rho3_NW = numpy_to_cvxpy(perm_rho3_NW)
    perm_rho3_SE = numpy_to_cvxpy(perm_rho3_SE)
    perm_rho3_SW = numpy_to_cvxpy(perm_rho3_SW)
    perm_rho3_list.append(perm_rho3_NE)
    perm_rho3_list.append(perm_rho3_NW)
    perm_rho3_list.append(perm_rho3_SE)
    perm_rho3_list.append(perm_rho3_SW)

    print('finish initialize syspermute of rho3')
    '''
    define contraints
    '''
    constraints = [cp.trace(rho0) == 1,
                   ptr_34 == rho0,
                   ptr_12 == rho0,
                   ptr_13 == rho0,
                   ptr_24 == rho0,
                   ptr_rho3_NE == rho2,
                   ptr_rho3_NW == rho2,
                   ptr_rho3_SE == rho2,
                   ptr_rho3_SW == rho2,
                   ptr_omega4_NE == V @ perm_rho3_NE @ V.T,
                   ptr_omega4_NW == V @ perm_rho3_NW @ V.T,
                   ptr_omega4_SE == V @ perm_rho3_SE @ V.T,
                   ptr_omega4_SW == V @ perm_rho3_SW @ V.T
                  ]
    #print('constraints ', constraints)

    # form objective
    obj = cp.Minimize(cp.trace(rho0@ham))
    #obj = cp.Minimize(cp.trace(rho_N_minus_1 @ sparse.kron(ham, sparse.identity(dim**(N-3)))))


    # form and solve problem
    prob = cp.Problem(obj, constraints)
    #prob.solve(solver=cp.MOSEK, verbose=True)
    #prob.solve(solver=cp.MOSEK)
    '''
    use the SCS solver for 1 iteration
    '''
    prob.solve(solver=cp.SCS, verbose=True, max_iters=1)
    #prob.solve(verbose=True)
    #if output==True:
    print("\nHuman readable SDP with cg on a 2D system")
    print("\nParamters:")
    #print("\nN =", N)
    print("dim =", dim)
    print("chi =", chi)
    print("\nRESULTS:\n")
    print("status:", prob.status)
    print("optimal value", prob.value)
    print("for cg map\n", W)

    #print("optimal var", ptr_R.value)
    #print(cp.trace((rho2@ham)).value)
    #print(rhos[0].value)
    '''
    print('rhos[0].value\n', rhos[0].value)
    print('\nrhos[1].value\n', rhos[1].value)
    print('\nrhos[2].value\n', rhos[2].value)
    print('\nomega.value\n', omega.value)
    print('\n rho_cg_L.value\n', rho_cg_L.value)
    '''
    print("-------------------------------------------------------")
    return prob, constraints, perm_rho3_list


def calc_norm_cg_rho(rho, W, dim=2):
    id = np.identity(dim)
    M = np.kron(W, id) @ rho @ np.kron(W.conj().T, id)
    norm = np.linalg.norm(M)
    #print('norm of corase grained rho:', norm)
    return norm


'''
sdp rewritten as
c^T * x = tr(rho * h)
constraints are in the form
sum x_i A_i
Bx = b
'''
def get_vectors_and_matrices(N, ham):
    #x_vec_cp = dsdp.Build_x_vector_cp(N)
    x_vec_cp = dsdp.Build_x_vector_cp(N)
    basis_vec = dsdp.set_basis_vector(N)
    A_mat = dsdp.Build_A_matrices(N, basis_vec)
    M_R_list, M_L_list = dsdp.Build_M_matrices(basis_vec)
    B_mat, b_vec = dsdp.Build_B_matrix_and_b_vector(M_R_list, M_L_list)
    c_vec = dsdp.Build_c_vector(ham, basis_vec, N)
    return c_vec, x_vec_cp, b_vec, A_mat, B_mat

def Minimize_cvec_timec_xvec(N, ham):
    # get all neede components
    c_vec, x_vec_cp, b_vec, A_mat, B_mat = get_vectors_and_matrices(N, ham)

    # define constraints
    constraints = []

    # sum x_i * A_i >= 0
    s = 0
    for i in range(0, (x_vec_cp.shape[0])):
        #print('i = ', i)
        s += x_vec_cp[i]*A_mat[i]
    print('s', s.shape)
    # >> is the matrix inequality
    constraints.append(s >> 0)

    # cast B times x into right shape to match with b
    constraints.append((B_mat@x_vec_cp).T[0] == b_vec)

    # form objective
    obj = cp.Minimize(c_vec@x_vec_cp)

    # form and solve problem
    prob = cp.Problem(obj, constraints)
    #prob.solve(solver=cp.SCS)
    #prob.solve(verbose=True)
    prob.solve()
    print("\n\nRESULTS:\n")
    print("WITHOUT coarse graining\n")
    print("status:", prob.status)
    print("optimal value", prob.value)
    #print("optimal var", x_vec_cp.value)
    #print('x_vec.shape', x_vec_cp.value.shape)
    print((c_vec@x_vec_cp).value)
    print("-------------------------------------------------------")
    return prob
