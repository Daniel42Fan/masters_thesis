'''
this file contains routines to calculate the SDP to return
the calculated energies
'''

import numpy as np
import cvxpy as cp
from scipy.io import loadmat
from heapq import nsmallest
from random import random
import pickle as pickle
import itertools as it
import os
import time
import sys
from datetime import timedelta
import tensornetwork as tn
import qutipy.general_functions as qt_gf
from qutipy.misc import cvxpy_to_numpy, numpy_to_cvxpy

import utility as ut
import solve_SDP as ssdp
import solve_SDP_cg as ssdp_cg
import define_SDP as dsdp
import define_SDP_cg as dsdp_cg
import define_system as ds
import analyze_data as ad


'''
create certain path, if needed
'''
def create_path(path_name):
    if not os.path.exists(path_name):
       os.makedirs(path_name)
    return path_name

'''
save stuff as pickle file
'''
def save_as_pickle(path, data_name, data):
    with open(path+data_name+'.pickle', 'wb') as f:
        pickle.dump(data, f)
    return None

'''
testing the behavior of converting cvxpy variable into numpy before
optimization is carried out
try use the partial trace function from the qutipy package
'''
def cp_var_into_np_var(ham, rhos, N=4):
    '''
    copied code from solve_SDP from the
    Minimize_trace_rho_times_h() function
    '''
    # system size
    # N
    rho2 = rhos[0]
    # create constraints
    constraints = []

    constraints.append(cp.trace(rho2) == 1)
    for i in range(0, len(rhos)-1):
        rho_shape_0 = rhos[i+1].shape[0]
        '''
        HERE: USE THE ptrace function with np, by converting the cvxpy var into
        a numpy var.
        Then convert the numpy result back into a cvxpy var.
        '''
        rhos[i+1] = cvxpy_to_numpy(rhos[i+1])
        print('timing using np ptr')
        t0 = time.time()
        ptr_R = ut.ptrace(rhos[i+1], [int(rho_shape_0/2), 2], 2)
        ptr_L = ut.ptrace(rhos[i+1], [2, int(rho_shape_0/2)], 1)
        t1 = time.time()
        total_time = t1-t0
        print('total time using np ptr =', timedelta(seconds=total_time))
        # ptr are now np object, do back conversion
        ptr_R = numpy_to_cvxpy(ptr_R)
        ptr_L = numpy_to_cvxpy(ptr_L)
        # also convert the rhos back
        rhos[i+1] = numpy_to_cvxpy(rhos[i+1])

        constraints.append(ptr_R == rhos[i])
        constraints.append(ptr_L == rhos[i])

    # form objective
    obj = cp.Minimize(cp.trace(rho2@ham))

    # form and solve problem
    prob = cp.Problem(obj, constraints)
    #prob.solve(solver=cp.SCS)
    #prob.solve(verbose=True)
    prob.solve()
    print("\nHuman readable SDP NO cg using NUMPY ptrace")
    print('N =', N)
    print("\nRESULTS:\n")
    print("status:", prob.status)
    print("optimal value", prob.value)

    print("-------------------------------------------------------")

    '''
    compare solutions with the original function
    '''
    prob2 = ssdp.Minimize_trace_rho_times_h(ham, rhos, N=4)
    return prob, prob2

'''
use np trace and cp trace of the 2D rho3 and compare the time
'''
def profiling_cp_vs_np_ptrace(dim=2):
    np.random.seed(1)
    mat = np.random.randint(10, size=(dim**(3*3), dim**(3*3)))
    print(mat.shape)
    rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    '''
    now using cp trace
    '''
    print('timing using cp ptr')
    t0 = time.time()
    # tracing out first subsystem
    ptr_rho3_cp1 = cp.partial_trace(rho3, list(np.ones(9, dtype='int')*dim), axis=0)
    # tracing out second subsystem
    ptr_rho3_cp2 = cp.partial_trace(ptr_rho3_cp1, list(np.ones(8, dtype='int')*dim), axis=0)
    # tracing out third subsystem
    ptr_rho3_cp3 = cp.partial_trace(ptr_rho3_cp2, list(np.ones(7, dtype='int')*dim), axis=0)
    # tracing out sixth subsystem
    ptr_rho3_cp6 = cp.partial_trace(ptr_rho3_cp3, list(np.ones(6, dtype='int')*dim), axis=2)
    # tracing out ninth subsystem
    ptr_rho3_cp9 = cp.partial_trace(ptr_rho3_cp6, list(np.ones(5, dtype='int')*dim), axis=4)
    t1 = time.time()
    total_time = t1-t0
    print('total time using cp ptr =', timedelta(seconds=total_time))
    # the final ptr is the last calculation step
    ptr_rho3_cp = ptr_rho3_cp9
    '''
    mat_ = numpy_to_cvxpy(mat)
    ptr_mat_cp1 = cp.partial_trace(mat_, list(np.ones(9, dtype='int')*dim), axis=0)
    # tracing out second subsystem
    ptr_mat_cp2 = cp.partial_trace(ptr_mat_cp1, list(np.ones(8, dtype='int')*dim), axis=0)
    # tracing out third subsystem
    ptr_mat_cp3 = cp.partial_trace(ptr_mat_cp2, list(np.ones(7, dtype='int')*dim), axis=0)
    # tracing out sixth subsystem
    ptr_mat_cp6 = cp.partial_trace(ptr_mat_cp3, list(np.ones(6, dtype='int')*dim), axis=2)
    # tracing out ninth subsystem
    ptr_mat_cp9 = cp.partial_trace(ptr_mat_cp6, list(np.ones(5, dtype='int')*dim), axis=4)
    '''
    '''
    using np trace, by converting cp into np
    '''
    print('timing using np ptr')
    t0 = time.time()
    ptr_rho3_NE_np = qt_gf.partial_trace(rho3, [1,2,3,6,9], list(np.ones(9, dtype='int')*dim))
    t1 = time.time()
    total_time = t1-t0
    print('total time using np ptr =', timedelta(seconds=total_time))
    ptr_mat_np = qt_gf.partial_trace(mat, [1,2,3,6,9], list(np.ones(9, dtype='int')*dim))
    '''
    ptr_mat_cp = cvxpy_to_numpy(ptr_mat_cp9)
    print('result using cp ptr for random matrix =', ptr_mat_cp)
    print('result using np ptr for random matrix =', ptr_mat_np)
    '''
    return 0

def profiling_syspermute(dim=2):
    rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    print('convert cp to np')
    rho3 = cvxpy_to_numpy(rho3)
    print('finish convert cp to np')
    print('calc syspermute')
    perm_rho3_NE = qt_gf.syspermute(rho3, [1,4,7,8,9,2,3,5,6], list(np.ones(9, dtype='int')*dim))
    print('finish calc syspermute')
    print('timing converting np to cp')
    t0 = time.time()
    rho3 = numpy_to_cvxpy(rho3)
    t1 = time.time()
    total_time = t1-t0
    print('total time converting np to cp =', timedelta(seconds=total_time))
    return 0

def profiling_cp_to_np_converion(dim=2):
    rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    chi = 1
    omega4 = cp.Variable((dim**(12)*chi, dim**(12)*chi), PSD=True)
    cvx_obj = rho3
    if cvx_obj.is_scalar():
        return np.array(cvx_obj)
    elif len(cvx_obj.shape) == 1:  # cvx_obj is a (column or row) vector
        return np.array(list(cvx_obj))
    else:  # cvx_obj is a matrix
        X = []
        for i in range(cvx_obj.shape[0]):
            x = [cvx_obj[i, j] for j in range(cvx_obj.shape[1])]
            X.append(x)
        X = np.array(X)
        return X

def cp_to_np_parallel(cvx_obj, it):
    #dim = 2
    #rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    #cvx_obj = rho3
    if cvx_obj.is_scalar():
        return np.array(cvx_obj)
    elif len(cvx_obj.shape) == 1:  # cvx_obj is a (column or row) vector
        return np.array(list(cvx_obj))
    else:  # cvx_obj is a matrix
        X = []
        X.append([cvx_obj[it, j] for j in range(cvx_obj.shape[1])])
        return np.array(X)

def cp_to_np_parallel2(it):
    dim = 2
    rho3 = cp.Variable((dim**(3*3), dim**(3*3)), PSD=True)
    cvx_obj = rho3
    if cvx_obj.is_scalar():
        return np.array(cvx_obj)
    elif len(cvx_obj.shape) == 1:  # cvx_obj is a (column or row) vector
        return np.array(list(cvx_obj))
    else:  # cvx_obj is a matrix
        X = []
        X.append([cvx_obj[it, j] for j in range(cvx_obj.shape[1])])
        return 0#np.array(X)

'''
function to calculate partial traces of multiple subsystems for a given
state using cvxpy
'''
def partial_trace_cp(mat, sys, dims):
    '''
    sys is a list of system
    dims is a list of dimension of each subsystem
    '''
    '''
    make a copy of the system and dim
    '''
    mat = mat
    #print(dims)
    sys_ = list(np.arange(1, len(dims)+1))
    for s in sys:
        #print('taking ptr of following systems:', sys)
        #print('system order:', sys_)
        s_idx = sys_.index(s)
        #print('ptr subsystem', s)
        #print('at index', s_idx)
        mat = cp.partial_trace(mat, dims, axis=s_idx)
        '''
        remove list entry of dim and sys_ at s_idx, since that system is now
        traced out
        '''
        del sys_[s_idx]
        del dims[s_idx]
        #print('sys_ after deleting element', sys_)
        #print('dim after deleting element', dims)
    return mat

'''
given a tensor (here: 3D matrix), contract it N times to receive a MPS
use ncon from the tensornetwork module
'''
def tensor_into_MPS(mat, N):
    if N>1:
        # get the matrix in a list N times
        m = list(it.repeat(mat, N))

        # create list with indices
        first = [(-1, -2, 1)]
        last = [(N-1, -(N+1), -(N+2))]
        middle = [(i, -(i+2), i+1) for i in range(1, N-1)]
        idx_list = first+middle+last
        #print(idx_list)
        # perform tensor contraction
        res = tn.ncon(m, idx_list)
        return res
    elif N==1:
        return mat
    else:
        return print('Index N must be >= 1.')

def tensor_list_into_MPS(list):
    # get the length of the list
    N = len(list)

    # create list with indices
    first = [(-1, -2, 1)]
    last = [(N-1, -(N+1), -(N+2))]
    middle = [(i, -(i+2), i+1) for i in range(1, N-1)]
    idx_list = first+middle+last
    #print(idx_list)
    # perform tensor contraction
    res = tn.ncon(list, idx_list)
    return res


'''
calculate the kron product recursively to get
|+000...0> and |-111...1>
'''
def calc_kron_rec(mat1, mat2, N, *arg):
    if N<1:
        print('Error: recursion can go any further')
    if N==1:
        res = np.kron(mat1, mat2)
        return res
    else:
        res = np.kron(mat1, mat2)
        N -= 1
        # if argument is empty, just get the kron prod of the 2nd matrix to the right
        if not arg:
            return calc_kron_rec(res, mat2, N)
        # otherwise get the kron prod the first matrix to the left
        else:
            return calc_kron_rec(mat1, res, N, arg)
'''
check if the MPS is implemented correctly
contract a Hadmard as follows:
-A-A-A-A-...A-
 | | | |    |
 H
 |
while A is contracted N times
choose A_0 = ((1,0), (0,0)), A_1 = ((0,0), (0,1))
So A = array([A_0, A_1])
(A is now a 3D matrix)
choose 2 random vectors u, v
contract them to
u-A-A-A-A-...A-v
  | | | |    |
  H
  |
The corresponding wavefunction is (for N=2)
psi = sum_ij (u_dagger A^i A^j v)ket(i)\otimes ket(j)

the result after contracting should be
psi = u0v0*ket(+0) + u1v1*ket(-0)

arguments:
N: number of additional As that are acting on -A-
                                               |
                                               H
pos: The Hadamard gate can be applied on any A, depending on the pos variable.
     By default it is acting on the 0th A.
'''
def check_MPS(N, pos=0):
    h = 1/np.sqrt(2)*np.array([[1,1], [1,-1]])
    #print('Hadamard =\n', h)
    basis_0 = np.array([1,0])
    basis_1 = np.array([0,1])

    # get plus and minus basis
    plus = h@basis_0
    minus = h@basis_1

    '''
    get the basis |000+0...0> and |111-1...1> for an arbritrary position of
    + or -
    '''
    basis_plus_0 = 0
    basis_minus_1 = 0
    if pos<=0:
        # get the |+000...0> and |-111...1> basis
        # N is # of 0 or 1
        plus_0 = calc_kron_rec(plus, basis_0, N)
        minus_1 = calc_kron_rec(minus, basis_1, N)
        basis_plus_0 = plus_0
        basis_minus_1 = minus_1
        if pos<0:
            print('Position where the H should act on must be >=0')
            print('Set the position to default at pos=0')
    elif pos>0:
        if N==pos:
            basis_plus_0 = calc_kron_rec(basis_0, plus, pos)
            basis_minus_1 = calc_kron_rec(basis_1, minus, pos)
        elif N>pos:
            plus_0 = calc_kron_rec(plus, basis_0, N-pos)
            basis_plus_0 = calc_kron_rec(basis_0, plus_0, pos, 'left')

            minus_1 = calc_kron_rec(minus, basis_1, N-pos)
            basis_minus_1 = calc_kron_rec(basis_1, minus_1, pos, 'left')
        if N<pos:
            print('Position must be somewhere within range.')
            print('Set position at default pos=0')
            # copied code, maybe can do it better...
            plus_0 = calc_kron_rec(plus, basis_0, N)
            minus_1 = calc_kron_rec(minus, basis_1, N)
            basis_plus_0 = plus_0
            basis_minus_1 = minus_1
    # reshape the |+000...0> and |-111...1> into a (2,2,2,...,2) matrix
    shape = tuple(2*np.ones((N+1), dtype=int))
    #plus_0 = plus_0.reshape(shape)
    #minus_1 = minus_1.reshape(shape)

    '''
    calculate
    B = (-1) - A - (-3)
               |(1)
               H
               |(-2)
    '''
    A_0 = np.array([[1,0], [0,0]])
    A_1 = np.array([[0,0], [0,1]])
    A = np.array([A_0, A_1])
    print('A =\n', A)
    print('A.shape =', A.shape)
    # transpose the array dimension in a right way
    A = np.transpose(A, (1,0,2))
    print('A transposed =\n', A)

    B = tn.ncon([A, h], [(-1, 1, -3), (1, -2)])
    #print('B = \n', B)
    #print('B.shape =\n', B.shape)

    # create a list of N times A
    A_N = list(it.repeat(A, N))
    #print('A_N.shape = ', A_N.shape)

    # create list with matrices to be contracted
    l = 0
    if pos<=0:
        l = [B] + A_N
        print(l)
    elif pos>0:
        A_N.insert(pos, B)
        l = A_N
    # contract B now with A_N
    C = tensor_list_into_MPS(l)

    # set u and v vectors
    (u0, u1) = (1, 0)
    (v0, v1) = (1, 0)
    u = np.array([u0, u1])
    v = np.array([v0, v1])

    # calculate the wavefunction psi representing the MPS
    psi = u0*v0*basis_plus_0 + u1*v1*basis_minus_1
    #print('psi =\n', psi)
    #print('psi.shape =\n', psi.shape)

    # calculate the MPS using ncon again
    # first: get the indices
    C_idx = tuple([1] + [-i for i in range(1, (N+1)+1)] + [2])
    #print('C idx =', C_idx)
    res = tn.ncon([u, C, v], [(1,), C_idx, (2,)])
    #print(res)
    #print('res.shape =\n', res.shape)
    #print('res flatten =\n', res.flatten())

    # calculate the norm of psi and the MPS
    norm = np.linalg.norm(psi-res.flatten())
    return print('Norm of Psi-mps is {}.'.format(norm))

'''
get data from a matlab file
'''
def get_matlab_data(path, data_name):
    data = loadmat(path+data_name)
    #print(data)
    return data

'''
form the matlab file, get the tensor for a given model
and prepare a MPS
'''
def prepare_MPS_state(model, bond_dimension):
    path = 'C:/Users/fdani/github/master-thesis/matlab_files/'
    data_name = 'savedMPSwithSubLatRot.mat'
    data = get_matlab_data(path, data_name)
    '''
    the keys are:
    dict_keys(['__header__', '__version__', '__globals__', 'Dmax', 'H_save',
                'Eexact_save', 'vuMPS_save', 'Eupper_save', 'xi_save',
                'converged', 'models'])
    '''
    '''
    For accessing the tensors, using to construct a MPS:
    vuMPS_save is teh structure storing the data.
    Indices:
    [a][b][c][d][e][f][g]
    a: which model? models are in the order
        0: ['heisenberg2'], 1: ['heisenberg2U'], 2: ['TFI'], 3: ['XXZ'],
        4: ['XXZU'], 5: ['XY'], 6:['heisenberg3']
    b: bond dimension (chi). Range: 1-8 (or index 0-7)
    c, d: necessary tensors are stored in are list inside a list.
          Set [c][d] = [0][0] to get past these lists
    e: here, tensors are stored.
       A_left, A_right, C and 2 more matrices
       [0] index to get the A_left tensor
    f: actual arrays are stored in another list, access them via [0], [1],...
    g: want to extrat the whole array out of another list using [0]
    '''
    # get the bond dimension (chi) index. Indexing starts from 0, so subtract 1
    chi_idx = bond_dimension-1
    models = {'heisenberg2': 0,
              'heisenberg2U': 1,
              'TFI': 2,
              'XXZ': 3,
              'XXZU': 4,
              'XY': 5,
              'heisenberg3': 6
    }
    # get model index
    m_idx = models[model]
    try:
        #print(data['vuMPS_save'][m_idx][chi_idx][0][0][0])
        mat = []
        # number of matrices needed to get out off of an array construct
        n = len(data['vuMPS_save'][m_idx][chi_idx][0][0][0])
        for i in range(n):
            # retrieve the matrices
            mat.append(data['vuMPS_save'][m_idx][chi_idx][0][0][0][i][0])
        # create the final matrix
        A_left = np.array(mat)
        #print(A_left)
        #print(A_left.shape)
        # transpose to prepare for contraction
        A_left = np.transpose(A_left, (1,0,2))
        #print(A_left.shape)
        '''
        WHERE DOES N COME FROM
        '''
        N = 5
        mps = tensor_into_MPS(A_left, N)
        # contract tensors to MPS
        return mps
    except:
        print('No data found for these settings.')
        return 0


#def run_grad_descent(N, dim, chi):
'''
TO DO (to make code easier): Delta should be in args inside a dictionary
'''
def run_grad_descent(N, chi, ham, linear_map, *args, dim=2,
                        seed_start=0, seed_end=100, g_tol=1e-6, model='XXZ',
                        normalize=False):
    N = N
    k = N-2
    dim = dim
    chi = chi
    Delta = args[0]
    ham = ham

    # safe the energies into a list
    energies_before_gd = []
    energies_after_gd = []

    # initalize variables
    res = 0
    e = 0
    res_history = []
    s = []
    r = []
    history_list = []
    # for specific seeds there are errors with MOESK
    for i in range(seed_start, seed_end):
    #for i in range(0, 1):
        seed = i
        print('######################################################################')
        print('seed =', seed)
        # used to calc the eigvals, eigvecs
        #H = ds.Build_H_XXZ_full(N-2, dim=dim, Delta=Delta)
        #print('H\n', H)
        # initialize W and path
        W = 0
        path = ''
        if linear_map=='pertubation':
            W, path = linear_map_pertubation(ham, N, dim, chi, Delta, seed)
        elif linear_map=='n_lowest_states':
            # initialize and get the number of lowest energy states through
            # the optional argumen
            n_states = 0
            for arg in args:
                if isinstance(arg, dict):
                    n_states = arg['n_states']
                    #print('n_states =', n_states)
                elif isinstance(arg, dict)==False:
                    pass
                else:
                    print('Parameter n_states no found.')
                    print('Set to default n_states = 4.')
                    n_states = 4
            W, path = linear_map_n_lowest_states(ham, N, chi, Delta, n_states, seed)
        elif linear_map=='gs_and_es':
            W, path = linear_map_gs_and_es(ham, N, dim, chi, Delta, seed)
        elif linear_map=='comb_of_gs':
            W, path = linear_map_comb_of_gs(ham, N, dim, chi, Delta, seed)
        elif linear_map=='random':
            W, path = linear_map_random(N, dim, chi, Delta, seed, model, *args, normalize=normalize)
        else:
            print('Chosen linear map is not valid. Please chose a valid linear map.')
            print('Code execution will be stopped.')
            sys.exit()
        prob, rho, __ = ssdp.Minimize_trace_rho_times_h_cg(ham, W, N=N, dim=dim, chi=chi, output=False)
        #norm = ssdp.calc_norm_cg_rho(rho, W, dim=dim)
        #print('\nnorm', norm)
        try:
            res, history, g_tol = ssdp_cg.gradient_descent(N, W, ham, g_tol=g_tol, dim=dim, chi=chi)
            '''
            g_tol necessary for chi=/=1 case!
            Comment out, if no more investigation of the gtol parameter is necessary
            '''
            #if chi!=1:
            #    path = path + 'g_tol{}/'.format(g_tol)
            optimized_cg_map = res.x
            optimized_cg_map = optimized_cg_map.reshape((chi, int(dim**k)))
            r.append(optimized_cg_map)
            res_history.append(res)
            # decide if the energies should be normalized or not
            if normalize==True:
                e, shift, norm_factor = ssdp.compare_normalized_solutions(N-1, N, prob.value, -res.fun, ham, dim=dim)
                # energies e: index 2 and 3 corresponds to before and after cg
                energies_before_gd.append(e[2])
                energies_after_gd.append(e[3])
                history_normalized = [(-1*h-shift)*norm_factor for h in history]
                history_list.append(history_normalized)
            else:
                energies_before_gd.append(prob.value)
                energies_after_gd.append(-res.fun)
                history_list.append(history)
        except Exception as err:
            print('Error at seed {}.'.format(seed))
            print(repr(err))
            s.append([seed, repr(err)])
        print('######################################################################')
    return energies_before_gd, energies_after_gd, s, r, history_list, res_history, path

def execute_calc_energies(dim=2):
    #g_tol = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    g_tol = [1e-8]
    linear_map = 'random'
    #linear_map = 'pertubation'
    #linear_map = 'n_lowest_states'
    #linear_map = 'gs_and_es'
    #linear_map = 'comb_of_gs'
    model = 'random_Hamiltonian'
    #model = 'XXZ'
    '''
    use random generated Hamiltonian
    add 1 to seed so that W and H dont share correlation
    '''
    ham_seeds = [3,6,9]
    #ham_seeds = [9]
    #for seed_ham in range(2, 3):
    for seed_ham in ham_seeds:
        ham = ds.Build_random_H(seed_ham)
        for d in range(2, 3):
        #for d in range(1,2):
            t0 = time.time()
            #ham = ds.Build_H_TFIM_Interaction_Term()
            #print('d =', d)
            Delta = d
            #ham = ds.Build_H_XXZ(Delta=Delta)

            #for N in range(7, 9, 2):
            for N in range(5, 7, 2):
                #print('N = ', N)
                for chi in range(2, 5):
                #for chi in range(2, 3):
                    #print('chi =', chi)
                    for g in g_tol:
                        #energies_before_gd, energies_after_gd, seed = run_grad_descent(N, 2, chi, ham, Delta)
                        #energies_before_gd, energies_after_gd, seed, W_final, history_list, path = run_grad_descent(N,
                        #                                                        chi, ham, linear_map, Delta, dim=2, seed_end=100)
                        energies_before_gd, energies_after_gd, seed, W_final, history_list, res_history, path = run_grad_descent(
                                N, chi, ham, linear_map, Delta, {'n_states': 4, 'seed_Hamiltonian': seed_ham},
                                dim=2, seed_end=100, g_tol = g, model=model, normalize=False)
                        print('before gd ', energies_before_gd)
                        print('after gd ', energies_after_gd)
                        # create path to save stuff
                        create_path(path)
                        save_as_pickle(path, 'energies_before_gd', energies_before_gd)
                        save_as_pickle(path, 'energies_after_gd', energies_after_gd)
                        save_as_pickle(path, 'optimized_cg_map', W_final)
                        save_as_pickle(path, 'history', history_list)
                        save_as_pickle(path, 'res_history', res_history)
                        with open(path+'error.txt', 'a+') as f:
                            print('Errors with following seeds:', seed)
                            for s in seed:
                                f.write('Error with seed {}.\n'.format(s[0]))
                                f.write('Message {}.\n'.format(s[1]))
            t1 = time.time()
            total_time = t1-t0
            print('total time for Delta = {}, N = {}:'.format(Delta, N), timedelta(seconds=total_time))
    return 0

def calc_energies_2D(ham, W, dim=2, chi=1):
    prob, constraints = ssdp.Minimize_trace_rho_times_h_cg_2D(ham, W, dim=dim, chi=chi)
    return prob, constraints


'''
Define different kinds of linear maps
Return the linear map and also a path, where the data can be stored afterwards
'''
# random generated linear map
def linear_map_random(N, dim, chi, Delta, seed, model, *args, normalize=False):
    W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed)
    model_0 = model
    # if args is not empty, get additional information about model
    # e.g. for random Hamiltonian get the seed information
    '''
    maybe change the if else statements to check which model is used
    and not what is in args
    '''
    if not args:
        pass
    else:
        for arg in args:
            if isinstance(arg, dict):
                '''
                NOTE THIS SEED IS DIFFERENT FROM THE SEED FOR W
                '''
                try:
                    s = arg['seed_Hamiltonian']
                    print('seed for Hamiltonian =', s)
                    model = model + '/seed' + str(s)
                except:
                    # if paramter seed_Hamiltonian is not included,
                    # then a different model was used
                    pass
            elif isinstance(arg, dict)==False:
                pass
            else:
                print('Parameter seed no found.')
                print('Code execution will be stopped.')
                sys.exit()
    path_1 = ad.get_data_path(model, normalize=normalize)
    path = path_1+'/random_maps/Delta{}_N{}_chi{}/'.format(Delta, N, chi)
    if model_0=='random_Hamiltonian':
        path = path_1+'/random_maps/N{}_chi{}/'.format(N, chi)
    else:
        pass
    #path = 'test/data/random_Hamiltonian/random_maps/seed{}_N{}_chi{}/'.format(Delta, N, chi)
    return W, path



'''
The maps down there are not actively used right now.
Path needs to be optimized using the ad.get_data_path() function
'''
# use the ground state with some small pertubation around it, specified with a
# parameter epsilon
def linear_map_pertubation(H, N, dim, chi, Delta, seed):
    eigvals, eigvecs = ut.eigen(H, chi)
    #get some pertubation
    epsilon = 0.1
    W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed)
    W = (eigvecs.T + epsilon*W)
    path = 'test/data/XXZ/pertubation/Delta{}_N{}_chi{}_epsilon{}/'.format(Delta, N, chi, epsilon)
    return W, path


'''
INSTEAD OF DISTINGUISH GS AND ES OR ONLY DEGENERATE GS, JUST USE A
LINEAR COMBINATION OF THE 3-4 LOWEST STATES.
'''
def linear_map_n_lowest_states(H, N, chi, Delta, n_states, seed):
    # get the eigvals and eigvecs of the n lowest states
    eigvals, eigvecs = ut.eigen(H, n_states)
    eigvecs_list = [eigvecs[:,i] for i in range(len(eigvecs[0]))]
    np.random.seed(seed)
    # for floats in range [a,b)
    a = -1
    b = 1
    random_numbers = (b - a)  * np.random.random_sample((len(eigvecs[0]),)) + a

    # get a linear combination of the n lowest states with random coefficients
    W = np.asarray(random_numbers)@np.asarray(eigvecs_list)

    path = 'test/data/XXZ/{}_lowest_states/Delta{}_N{}_chi{}/'.format(n_states, Delta, N, chi)
    return W, path


# use a linear combination of ground state and the first excited state
def linear_map_gs_and_es(H, N, dim, chi, Delta, seed):
    eigvals, eigvecs = ut.eigen(H, chi)
    # round to avoid numerical errors
    eigvals = np.round(eigvals, decimals=14)
    # get the smallest eigenvalue = ground state
    gs = nsmallest(1, eigvals)
    # the first excited state is the 2nd smallest element of all eigenvalues
    # (after the ground state)
    es = nsmallest(2, eigvals)[-1] # last element is the 2nd smallest

    # get the indices of the gs/es
    gs_idx = [i for i, x in enumerate(eigvals) if x == gs]
    es_idx = [i for i, x in enumerate(eigvals) if x == es]
    # superposition of the needed eigvecs
    W = eigvecs[:,gs_idx[0]] + eigvecs[:,es_idx[0]]

    path = ''
    return W, path

# since the Hamiltonian is degenerate, use  a combination of the ground states
# with random normalized coefficients.
def linear_map_comb_of_gs(H, N, dim, chi, Delta, seed):
    eigvals, eigvecs = ut.eigen(H, chi)
    # round to avoid numerical errors
    eigvals = np.round(eigvals, decimals=10)
    gs = nsmallest(1, eigvals)
    # indices of all ground states
    gs_idx = [i for i, x in enumerate(eigvals) if x == gs]
    # get the eigenvectors of all degenerate groundstates
    gs_list = [eigvecs[:,gs_idx[i]] for i in range(len(gs_idx))]
    # generate random numbers
    random_numbers = []
    for i in range(len(gs_list)):
        random_numbers.append(random())
    # normalize the random numbers, so their sum is = 1
    #s = sum(random_numbers) # sum of square!!!
    #random_numbers_normalized = [i/s for i in random_numbers]
    '''
    c are the random coefficients
    v are eigenvectors
    W = sum_i c_i v_i
    '''
    W = np.asarray(random_numbers)@np.asarray(gs_list)
    path = ''
    return W, path
