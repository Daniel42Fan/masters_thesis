"""
This module contains functions (and classes) that help define our
semi-definite problem with cg
"""
import numpy as np
from numpy import linalg as LA
import cvxpy as cp
import scipy.sparse as sparse
from scipy.linalg import block_diag
from scipy.optimize import approx_fprime, check_grad

import utility as ut
import define_SDP as dsdp
import solve_SDP as ssdp
import solve_SDP_cg as ssdp_cg
import define_SDP_cg as dsdp_cg

# construct the basis needed in the problem
def set_basis_vector_cg(n_cols, n_rows):
    # store the basis into a vector
    basis_vec = []

    # create a set of orthonormal basis e.g. in 2D
    # {[[1,0], [0,0]],[[0,1], [0,0]],  [[0,0], [1,0]], [[0,0], [0,1]]}
    # and so on for higher dimensions
    for i in range(n_cols):
        for j in range(n_rows):
            # create a matrix with right dimensions with only 0 as entries
            temp_matrix = np.zeros((n_cols, n_rows))
            # fill the i,j-th entry with a single 1
            temp_matrix[i][j] = 1
            # sparse it
            '''
            Comment out to get a better view vor testing
            '''
            #temp_matrix = sparse.csr_matrix(temp_matrix)
            basis_vec.append(temp_matrix)

    return basis_vec

def set_variable_dimensions(N, dim=2, chi=3):
    '''
    For later:
    maybe build a dictinary with rho2: dim d^2 x d^2
    and so on
    '''
    d = dim
    chi = chi

    rho_dims = [(d**i * d**i) for i in range(2,4)]
    omega_dims = [(d**2 * chi * d**2 * chi) for i in range(4, N+1)]
    return rho_dims, omega_dims

'''
defines basis vectors neede for the A matrix
'''
def get_basis_vector_cg(N, dim=2, chi=3):
    # since the rhos and omegas are quadratic, n_cols = n_rows
    n_cols = n_rows = []

    rho_dims, omega_dims = set_variable_dimensions(N, dim=dim, chi=chi)
    # take square root, since we want the actual m of the mxm matrices
    rho_dims = np.sqrt(rho_dims)
    omega_dims = np.sqrt(omega_dims)

    for dim in rho_dims:
        n_cols.append(int(dim))
    for dim in omega_dims:
        n_cols.append(int(dim))
    n_rows = n_cols
    #print('n_rows', n_rows)
    basis_vec = []

    for n in n_rows:
        basis_vec.append(set_basis_vector_cg(n, n))

    return basis_vec

# get x vector as cp Variable WITH cg
def Build_x_vector_cp_cg(N, dim=2, chi=3):
    '''
    x has all the coeff. of the (rho_2, rho_3, omega_4, omega_5,..., omega_n)
    The omegas depend on the dimension d and bond dimension chi
    '''
    '''
    # former function
    # get length of the vector
    l = sum([(2**i)*(2**i) for i in range(2, N+1)])
    x_vec_cp = cp.Variable((l, 1))
    '''
    d = dim
    chi = chi

    # get the number of coefficients for rho_2 and rho_3
    l_rho = sum([(d**i)*(d**i) for i in range(2, 4)])

    # we need to prepare N-3 of the omegas
    # all omegas have dimension (chi*d^2) x (chi*d^2)
    # get the length of the vector (number of coefficients) for all the omegas
    l_omega = sum([(chi*(d**2))*(chi*(d**2)) for i in range(0, N-3)])

    # total length fo x_vector
    l = l_rho + l_omega
    #print('len x vector', l)
    x_vec_cp = cp.Variable((l, 1))
    return x_vec_cp


'''
build c vector
needs as parameter the Hamiltonian + basis vector
'''
def Build_c_vector_cg(N, ham, basis_vec, dim=2, chi=3):
    d = dim
    chi = chi
    c_vec = []
    # total length of c vec
    l_rho = sum([(d**i)*(d**i) for i in range(2, 4)])
    l_omega = sum([(chi*(d**2))*(chi*(d**2)) for i in range(0, N-3)])
    total_length = l_rho + l_omega

    # for the c vector, we only need the first 16 components out of the
    # x-vector and basis vector, since
    # c^T * x = tr(rho^(2) * h)
    # depends on rho^(2), which only has 16 components

    ind = 16
    for i in range(0, ind):
        c_vec.append(np.trace((basis_vec[0][i].T)@ham))

    for i in range(ind, total_length):
        c_vec.append(0)

    return c_vec

'''
builds the A matrix
only change is the A_dims and the dim, chi parameter
'''
def Build_A_matrices_cg(N, basis_vec, dim=2, chi=3):
    A_vec = []
    # need the dimension of the A matrix
    rho_dims, omega_dims = set_variable_dimensions(N, dim=dim, chi=chi)
    A_dims = []
    for d in rho_dims:
        A_dims.append(int(d))
    for d in omega_dims:
        A_dims.append(int(d))
    #print('A_dims', A_dims)

    basis_vec = tuple(basis_vec)
    #print('basis_vec = ', basis_vec)


    # A matrix is the form
    # A^1_i = [[E^1_i, 0, 0], [0, 0, 0], [0, 0, 0]]
    # A^2_i = [[0, 0, 0], [0, E^2_i, 0], [0, 0, 0]]
    # etc... in the appropiate dimension

    # depending on how many different sizes of basis we have
    # (e.g. 4x4, 16x16, 64x64 are three basis with different size)
    #
    zero_matrices = []
    for j in range(0, len(A_dims)):
        #print(j)
        # create all neccessary zero matrices that are used in the block matrix
        zero_matrix = np.zeros((int(np.sqrt(A_dims[j])), int(np.sqrt(A_dims[j]))))
        zero_matrices.append(zero_matrix)
    #print('zero', zero_matrices)

    #A_mat = zero_matrices
    #print('A_mat = ', A_mat)
    for k in range(0, len(A_dims)):
        #print('k ', k)
        #print('basis_vec[k] = ', basis_vec[k])
        for l in range(A_dims[k]):
            #print('l ', l)
            #print('basis_vec[k][l] = ', basis_vec[k][l])
            #print('ind ', ind)

            # replace a zero matrix with an actual basis, so this list consists
            # of the diagonal components of the desired block matrix
            A_mat = zero_matrices.copy()

            #print(basis_vec[k][l])
            A_mat[k] = basis_vec[k][l]
            #print('A_mat[k] = ', A_mat[k])
            #print('A-mat', A_mat)

            A_vec.append(A_mat)


    #print(A_vec[0])
    #print(len(A_vec))
    # get the block matrices
    A_matrices_final = []
    for i in range(len(A_vec)):
        res = list(map(block_diag, *zip(tuple(A_vec[i]))))[0]
        A_matrices_final.append(res)

        #IMPLEMENT SPARSE MATRIX LATER

    return A_matrices_final


'''
THIS IS NOT SCALEABLE YET FOR N>4
'''
def get_V_matrices_old(N, dim=2, chi=3):
    # set dimensions
    d = dim
    chi = chi
    # store the V matrices in a list
    V_L_mat = []
    V_R_mat = []
    # get basis vectors: 3 = N-1, for N=4?
    n_cols_in = d**3
    n_rows_in = d**3
    basis_vec_in = set_basis_vector_cg(n_cols_in, n_rows_in)

    n_cols_out = chi*d
    n_rows_out = chi*d
    basis_vec_out = set_basis_vector_cg(n_cols_out, n_rows_out)

    '''
    NOT SCALABLE, SINCE WE NEED TO GET Vs WITH ITERATIVE CG
    '''
    lin_op = ut.linear_op_cg_maps
    V_L_mat.append(ut.Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'L', dim, chi))
    V_R_mat.append(ut.Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'R', dim, chi))
    return V_L_mat, V_R_mat

# for gradient descent
def get_V_matrices(N, W, dim=2, chi=3, grad_descent=False, index_tuple=(0,0), roll=0):
    # set dimensions
    d = dim
    chi = chi
    # get the cg map
    W = W
    # store the V matrices in a list
    V_L_mat = []
    V_R_mat = []
    # get basis vectors: 3 = N-1, for N=4?
    n_cols_in = d**3
    n_rows_in = d**3
    basis_vec_in = set_basis_vector_cg(n_cols_in, n_rows_in)

    n_cols_out = chi*d
    n_rows_out = chi*d
    basis_vec_out = set_basis_vector_cg(n_cols_out, n_rows_out)

    '''
    NOT SCALEABLE, SINCE WE NEED TO GET Vs WITH ITERATIVE CG
    '''
    lin_op = ut.linear_op_cg_maps
    if grad_descent==False:
        V_L_mat.append(ut.Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'L',
                                        dim, chi, W))
        V_R_mat.append(ut.Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'R',
                                        dim, chi, W))
        return V_L_mat, V_R_mat
    elif grad_descent==True:
        #print('inside this grad descent')
        V_L_mat.append(ut.Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'L',
                    dim, chi, W, grad_descent=grad_descent, index_tuple=index_tuple, roll=roll))
        V_R_mat.append(ut.Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'R',
                    dim, chi, W, grad_descent=grad_descent, index_tuple=index_tuple, roll=roll))
        return V_L_mat, V_R_mat

    return print('No V_mat matrix returned. Missing some arguments.')

'''
get E matrix used to calculate the derivative of V, where
E_L = E_kl \otimes id
E_R = id \otimes E_kl
for indices kl
'''
def get_E_matrices(N, index_tuple, dim=2, chi=3):
    # create basis matrix E_kl matrix

    m = chi
    n = int(dim**(N-2))
    s = (m,n)
    #print('s = ', s)
    # for the derivative one of the W needs to be replaced by a basis.
    basis = ut.create_basis_matrix(s, index_tuple)
    #print('basis.shape', basis.shape)

    id = sparse.identity(dim)
    #print('id.shape = ', id.shape)

    E_L = sparse.kron(basis, id)
    E_L_dagger = sparse.kron(basis.conj().T, id)

    E_R = sparse.kron(id, basis)
    E_R_dagger = sparse.kron(id, basis.conj().T)
    '''
    print('E_L = ', E_L)
    print('E_L.shape = ', E_L.shape)
    print('E_L_dagger.shape = ', E_L_dagger.shape)
    print('E_R.shape = ', E_R.shape)
    print('E_R_dagger.shape = ', E_R_dagger.shape)
    '''
    return E_L, E_L_dagger, E_R, E_R_dagger




'''
check, if the SDP gradient formula gives the right result
It should be:
              LHS = RHS
<y (d/dW_ij B) x> = d/dW_ij (tr( (W \otimes  Id)rho^3 (W_dagger \otimes  Id)gamma_L ) +
                            tr( (Id \otimes  W)rho^3 (Id \otimes  W_dagger)gamma_R ))
where gamma_L/R correspond to the dual variables of the constraints
    (W \otimes  Id)rho^3 (W_dagger \otimes  Id) =  tr_L(omega^4),
    (Id \otimes  W)rho^3 (Id \otimes  W_dagger) =  tr_R(omega^4)
'''

'''
get the energy and its derivative
de/dW
with the coarse graining map as (optimization) variable (for later)
'''
global global_energy_list
global_energy_list = []
def get_energy_and_derivative_of_energy(W, N, ham, dim=2, chi=3):
    # create index tuples for initializing the matrix dimension of the derivative
    # W has dimension chi x d^2
    # or chi x d^(N-2)
    k = chi #W.shape[0]
    #print('N', N)
    #print('dim', dim)
    l = int(dim**(N-2)) #W.shape[1]
    ham = ham
    rhos = ssdp.rhos_cp_variable(N, dim=dim)
    #prob, rho3, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, rhos, W, N=N, dim=dim, chi=chi)
    prob, rho_N_minus_1, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, W, N=N, dim=dim, chi=chi, output=False)
    '''
    if needed, see the prob status
    '''
    #print('status:', prob.status)
    # the gamma matrices correnspond to constraints[3] and constraints[4]
    # IF the problem is formulated in terms of rho2, rho3...
    '''
    gamma_R = constraints[3].dual_value
    gamma_L = constraints[4].dual_value
    '''
    gamma_R = constraints[2].dual_value
    gamma_L = constraints[3].dual_value

    # initialize variable for derivative
    derivative_L = 0
    derivative_R = 0

    # get the E_LR (dagger) and V_LR (daggger) for the derivative
    #V_L, V_L_dagger, V_R, V_R_dagger = cg_maps(W, dim=dim, chi=chi)
    V_L, V_L_dagger, V_R, V_R_dagger = cg_maps_N(W, N=N, dim=dim, chi=chi)

    indices = []
    # create all possible indices of the W matrix
    for i in range(0, k):
        for j in range(0, l):
            indices.append((i,j))

    # initialize matrix
    mat_L = np.zeros((k, l))
    mat_R = np.zeros((k, l))
    #mat_L_list = []
    #mat_R_list = []

    for index_tuple in indices:
        #print('############################################################')
        #print('INDEX_TUPLE', index_tuple)
        E_L, E_L_dagger, E_R, E_R_dagger = get_E_matrices(N, index_tuple, dim=dim,
                                                            chi=chi)
        (i,j) = index_tuple
        # use product rule
        # be aware of the minus sign, like in the SDP gradient paper
        #mat_L[i][j] = -1*(np.trace(E_L @ rho3 @ V_L_dagger @ gamma_L) + np.trace(V_L @ rho3 @ E_L_dagger @ gamma_L))
        mat_L[i][j] = -1*(np.trace(E_L @ rho_N_minus_1 @ V_L_dagger @ gamma_L) + np.trace(V_L @ rho_N_minus_1 @ E_L_dagger @ gamma_L))

        #mat_R[i][j] = -1*(np.trace(E_R @ rho3 @ V_R_dagger @ gamma_R) + np.trace(V_R @ rho3 @ E_R_dagger @ gamma_R))
        mat_R[i][j] = -1*(np.trace(E_R @ rho_N_minus_1 @ V_R_dagger @ gamma_R) + np.trace(V_R @ rho_N_minus_1 @ E_R_dagger @ gamma_R))
        #print(f'mat_L matrix at index {index_tuple}\n', mat_L)

    deriv = mat_L + mat_R
    #print(deriv)
    '''
    SINCE WE WANT THE MAXIMUM, WE NEED TO TRANSFORM THE RESULT AS
    - MIN(-FUNCTION)
    '''
    deriv = -1*deriv
    #print(deriv)
    deriv = deriv.flatten()
    #print(deriv.flatten())
    #print('deriv = ', deriv)
    #print('-prob.value =', -prob.value)

    '''
    to do:
    return the energy as global variable? so the callback function in the
    gradient descent can store these energies as interim findings
    '''
    global_energy_list.append(-prob.value)
    #print('global energy inside dsdp_cg', global_energy)

    return (-prob.value, deriv)

def get_energy(W, N, ham, dim=2, chi=3):
    # create index tuples for initializing the matrix dimension of the derivative
    # W has dimension chi x d^2
    # or chi x d^(N-2)
    ham = ham
    rhos = ssdp.rhos_cp_variable(N, dim=dim)
    #prob, rho3, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, rhos, W, N=N, dim=dim, chi=chi)
    prob, rho_N_minus_1, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, W, N=N, dim=dim, chi=chi, output=False)
    return -prob.value

def get_derivative_of_energy(W, N, ham, dim=2, chi=3):
    ham = ham
    rhos = ssdp.rhos_cp_variable(N, dim=dim)
    #prob, rho3, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, rhos, W, N=N, dim=dim, chi=chi)
    prob, rho_N_minus_1, constraints = ssdp.Minimize_trace_rho_times_h_cg(ham, W, N=N, dim=dim, chi=chi, output=False)
    k = chi #W.shape[0]
    l = int(dim**(N-2)) #W.shape[1]
    gamma_R = constraints[2].dual_value
    gamma_L = constraints[3].dual_value

    # initialize variable for derivative
    derivative_L = 0
    derivative_R = 0
    # get the E_LR (dagger) and V_LR (daggger) for the derivative
    V_L, V_L_dagger, V_R, V_R_dagger = cg_maps_N(W, N=N, dim=dim, chi=chi)

    indices = []
    # create all possible indices of the W matrix
    for i in range(0, k):
        for j in range(0, l):
            indices.append((i,j))

    # initialize matrix
    mat_L = np.zeros((k, l))
    mat_R = np.zeros((k, l))

    for index_tuple in indices:
        E_L, E_L_dagger, E_R, E_R_dagger = get_E_matrices(N, index_tuple, dim=dim,
                                                            chi=chi)
        (i,j) = index_tuple
        # use product rule
        # be aware of the minus sign, like in the SDP gradient paper
        mat_L[i][j] = -1*(np.trace(E_L @ rho_N_minus_1 @ V_L_dagger @ gamma_L) + np.trace(V_L @ rho_N_minus_1 @ E_L_dagger @ gamma_L))
        mat_R[i][j] = -1*(np.trace(E_R @ rho_N_minus_1 @ V_R_dagger @ gamma_R) + np.trace(V_R @ rho_N_minus_1 @ E_R_dagger @ gamma_R))

    deriv = mat_L + mat_R
    '''
    SINCE WE WANT THE MAXIMUM, WE NEED TO TRANSFORM THE RESULT AS
    - MIN(-FUNCTION)
    '''
    deriv = -1*deriv
    deriv = deriv.flatten()
    return deriv

'''
rewrite the get energy and derivative function by getting function value and
derivative separately
'''
def get_energy_and_its_derivative(W, N, ham, dim=2, chi=3):
    prob = get_energy(W, N, ham, dim=2, chi=3)
    deriv = get_derivative_of_energy(W, N, ham, dim=2, chi=3)
    return (-prob.value, deriv)


def get_energy_and_its_derivative_2D(W, ham, dim=2, chi=1):
    # create index tuples for initializing the matrix dimension of the derivative
    # W has dimension chi x d^4
    k = chi #W.shape[0]
    l = int(dim**4) #W.shape[1]
    ham = ham

    prob, constraints, perm_rho3_list = ssdp.Minimize_trace_rho_times_h_cg_2D(ham, W, dim=dim, chi=chi)
    # the gamma matrices correnspond to constraints[9] to constraints[12]
    gamma_NE = constraints[9].dual_value
    gamma_NW = constraints[10].dual_value
    gamma_SE = constraints[11].dual_value
    gamma_SW = constraints[12].dual_value

    # initialize variable for derivative
    derivative_NE = 0
    derivative_NW = 0
    derivative_SE = 0
    derivative_SW = 0

    # for N=6, k=N-2=4, for a (chi x d^k) map
    N=6
    W = generate_linear_map_N(N=N, dim=dim, chi=chi)
    V_L, V_L_dagger, V_R, V_R_dagger = cg_maps_N(W, N=N, dim=dim, chi=chi)

    indices = []
    # create all possible indices of the W matrix
    for i in range(0, k):
        for j in range(0, l):
            indices.append((i,j))

    # initialize matrix
    mat_NE = np.zeros((k, l))
    mat_NW = np.zeros((k, l))
    mat_SE = np.zeros((k, l))
    mat_SW = np.zeros((k, l))

    for index_tuple in indices:
        #print('############################################################')
        #print('INDEX_TUPLE', index_tuple)
        E_L, E_L_dagger, E_R, E_R_dagger = get_E_matrices(N, index_tuple, dim=dim,
                                                            chi=chi)
        (i,j) = index_tuple
        # use product rule
        mat_NE[i][j] = -2*(np.trace(gamma_NE @ V_R @ perm_rho3_list[0] @ E_R_dagger))
        mat_NW[i][j] = -2*(np.trace(gamma_NW @ V_R @ perm_rho3_list[1] @ E_R_dagger))
        mat_SE[i][j] = -2*(np.trace(gamma_SE @ V_R @ perm_rho3_list[2] @ E_R_dagger))
        mat_SW[i][j] = -2*(np.trace(gamma_SW @ V_R @ perm_rho3_list[3] @ E_R_dagger))


    '''
    is the derivative here also the sum of everything???
    deriv = mat_L + mat_R
    '''
    deriv = mat_NE + mat_NW + mat_SE + mat_SW
    #print(deriv)
    '''
    SINCE WE WANT THE MAXIMUM, WE NEED TO TRANSFORM THE RESULT AS
    - MIN(-FUNCTION)
    '''
    deriv = -1*deriv
    #print(deriv)
    deriv = deriv.flatten()
    #print(deriv.flatten())
    #print('deriv = ', deriv)
    return (-prob.value, deriv)

# Hamiltonian as an input variable
def check_SDP_grad_formula(N, W, ham, dim=2, chi=3):
    '''
    adapted from get_V_matrices, since the procedure is really similiar
    '''
    # get the cg map
    W = W
    '''
    # store the V matrices in a list
    V_L_mat = []
    V_R_mat = []
    '''
    # create index tuples for initializing the E, LHS and RHS matrices
    k = W.shape[0]
    l = W.shape[1]

    '''
    solve the SDP problem and get the rho3 matrix and gamma matrices
    '''
    ham = ham
    prob_2, optimal_variable, optimal_dual_variable, constraints_2 = ssdp_cg.Minimize_cvec_timec_xvec_cg(N, ham, W, dim=dim, chi=chi)
    # x_vec is the optimal variable
    x_vec = optimal_variable
    # y_vec correndponds to the second constraint stored in constraints_2[1]
    #y_vec = constraints_2[1].dual_value
    y_vec = optimal_dual_variable

    LHS_matrix = np.empty(shape=(k, l))
    #grad_B_kl = np.empty(shape=(k, l))
    # get gradient of B at the position (k,l)

    for k_ in range(0, k):
        for l_ in range(0, l):
            grad_B_kl = Build_B_matrix_cg(N, W, dim=dim, chi=chi, gradient=True, grad_kl=True, index_tuple=(k_,l_))
            # dont forget minus sign, like in the SDP gradient paper
            LHS_matrix[k_, l_] = - y_vec @ grad_B_kl @ x_vec

            print(k_, l_)

    # left hand side of the formula
    #print('LHS = ', LHS_matrix)

    _, RHS = get_energy_and_derivative_of_energy(W, N, ham, dim=2, chi=3)
    print('RHS = ', RHS)
    print('LHS = ', LHS_matrix)

    return print('norm of RHS-LHS = ', np.linalg.norm(RHS-LHS_matrix))

'''
use scipy.optimize.check_grad function to check to gradient computed by hand
with the result of scipy.optimize.approx_fprime
'''
def check_grad_function(W, N, ham, seed):
    dim = 2
    chi = 3
    #func = get_energy_and_derivative_of_energy(W, N, ham, dim=2, chi=3)
    points = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=seed)

    return check_grad(get_energy, get_derivative_of_energy, points.flatten(),
                N, ham, epsilon=1e-6)

'''
THIS IS NOT SCALEABLE YET FOR N>4
'''
def get_M_matrices(N, dim=2, chi=3):
    # set dimensions
    d = dim
    chi = chi
    # store the V matrices in a list
    M_L = []
    M_R = []
    # get basis vectors
    n_cols_in = d**2 * chi
    n_rows_in = d**2 * chi
    M_basis_vec_in = set_basis_vector_cg(n_cols_in, n_rows_in)

    # we trace out one dimension, so output is input_dim/ 1dim
    n_cols_out = d*chi
    n_rows_out = d*chi
    M_basis_vec_out = set_basis_vector_cg(n_cols_out, n_rows_out)

    # also get the M matrices WITHOUT CG
    # reuse some old code
    basis_vec = dsdp.set_basis_vector(4)
    M_R_list, M_L_list = dsdp.Build_M_matrices(basis_vec)
    # get the first element out of that list
    M_R.append(M_R_list[0])
    M_L.append(M_L_list[0])

    '''
    NOT SCALABLE, SINCE WE NEED TO GET Vs WITH ITERATIVE CG
    '''
    lin_op = ut.linear_op_ptr
    M_R.append(ut.Build_matrix_rep(lin_op, M_basis_vec_in, M_basis_vec_out, (dim*chi, dim), 2))
    M_L.append(ut.Build_matrix_rep(lin_op, M_basis_vec_in, M_basis_vec_out, (dim, dim*chi), 1))
    return M_L, M_R



'''
Build the B matrix
"gradient" parameter is used, if the gradient of B is needed (for gradient descent)
So we can reuse code.
'''
def Build_B_matrix_cg(N, W, dim=2, chi=3, gradient=False, grad_kl=False, index_tuple=(0,0)):
    # build the matrix that represents the trace
    def Build_T_matrix():
        T = np.zeros(shape=(1,16))
        for i in range(0, 16):
            if i%5==0:
                T[0][i] = 1
        return T

    T = Build_T_matrix()

    # initialize dimensions
    N = N
    d = dim
    chi = chi

    # get cg map
    W = W

    '''
    get the dimension of the variables (rho_2, rho_3, omega_4, ..., omega_N)
    '''
    # for rho_2 and rho_3
    var_dims = [(d**i)*(d**i) for i in range(2, 4)]
    # for the omegas
    var_dims += [(chi*d**2)*(chi*d**2) for i in range(4, N+1)]
    '''
    get the n dimensions of the mxn blockmatrices, depending on N
    since we used copied code (and do not want to destroy code from below)
    just assign var_dims to it
    '''

    block_dims = var_dims

    '''
    get the dimension of the matrices from the constraints
    V * rho_(N-1) * V_dagger = ptr(omega_N)
    after cg. Since omega_N have dimension chi*d^2 x chi*d^2,
    the ptr traces out the dimension d.
    '''
    dim_after_cg = chi**2 * d**2

    #print(block_dims)
    # copy the dimensions, to get the indices in an easier way
    # for the zeros matrices
    block_dims_roll = block_dims.copy()
    #print(block_dims_roll)

    '''
    get the V and M matrices
    '''
    V_L_list = 0
    V_R_list = 0
    dV_L_mat = 0
    dV_R_mat = 0
    if gradient==True:
        V_L_list, V_R_list, dV_L_mat, dV_R_mat = ut.get_derivative_V(N, W, dim=d, chi=chi)
    else:
        V_L_list, V_R_list = get_V_matrices(N, W, dim=d, chi=chi)
    M_L_list, M_R_list = get_M_matrices(N, dim=d, chi=chi)



    #print('M_L_list', M_L_list)
    #print('M_L_list[0]', M_L_list[0])
    # B is going to be a block matrix
    # It contains of (2N-3)x(N-1) blocks
    # initialize an empty array of that size
    B_temp = np.empty(shape=(2*N-3, N-1), dtype='object')

    # counter to track, how many times the first for loop is running
    # can be used as index to get the right dimension out of the block_dims
    dim_counter = 0
    # as well to track the indices for the M_lists
    M_counter = 0
    V_counter = 0
    # for loop i index: increment by 2
    # e.g. we get for N=4: i=0,2,4
    # so we can fill out i=0, then i=1,2 togehter and then i=3,4 components
    block_dims_roll_counter = 0

    # if we want dB/dW_kl for index kl, the specific dV/dW_kl needs to be inserted
    # get the k,l index
    k = W.shape[0]
    l = W.shape[1]

    for i in range(0, (2*N-3), 2):
        #print('i', i)
        for j in range(N-1):
            #print('j1', j)
            # first row of B
            # 1x16, 1x64, 1x256,.... etc matrix
            if (i==0 and j==0):
                '''
                build the trace matrix T (1x16)
                '''
                if gradient==True:
                    B_temp[i][j] = np.zeros_like(T)
                else:
                    B_temp[i][j] = T
            elif (i==0 and j!=0):
                '''
                block_dims has length N-1,
                so just use the j index to read out that list
                '''
                B_temp[i][j] = np.zeros(shape=(1, block_dims[j]))
                #B_temp[i][j] = sparse.csr_matrix(np.zeros(shape=(1, block_dims[j])))
                #B_temp[i][j] = id#'0'#sparse.csr_matrix(np.zeros(shape=(1, block_dims[j])))
            elif (i==2 and j==0):
                '''
                identity matrices, needed for rho_2 = ptr(rho_3)
                only for i == 2, since i >=3 has V matrices with the cg maps
                '''
                if gradient==True:
                    B_temp[i-1][j] = np.zeros((block_dims[dim_counter], block_dims[dim_counter]))
                    B_temp[i][j] = np.zeros((block_dims[dim_counter], block_dims[dim_counter]))
                else:
                    B_temp[i-1][j] = np.identity(block_dims[dim_counter])
                    #B_temp[i-1][j] = sparse.csr_matrix(np.identity(block_dims[dim_counter]))
                    B_temp[i][j] = np.identity(block_dims[dim_counter])
                    #B_temp[i][j] = sparse.csr_matrix(np.identity(block_dims[dim_counter]))
                    #B_temp[i][j] = id#'1'#sparse.csr_matrix(np.identity(block_dims[dim_counter]))
                # increment counter for next time, when the elif is satisfied
                '''
                since this elif will only saisfy 1 time, this line can be deleted
                Leave it here for now, bc it is copied code and
                everything seems like to work so far as it is
                '''
                dim_counter += 1
            elif (i!=0 and j==1):
                '''
                M matrices
                '''
                #print('M_counter', M_counter)
                if gradient==True:
                    B_temp[i-1][j] = np.zeros_like(M_L_list[M_counter])
                    B_temp[i][j] = np.zeros_like(M_R_list[M_counter])
                else:
                    B_temp[i-1][j] = -1*M_L_list[M_counter]
                    B_temp[i][j] = -1*M_R_list[M_counter]
                #B_temp[i-1][j] = id#'ML'#-1*M_L_list[M_counter]
                #B_temp[i][j] = id#'MR'#-1*M_R_list[M_counter]
                M_counter += 1
            elif (i>2 and j==0):
                if grad_kl==True:
                    (k,l) = index_tuple
                    B_temp[i-1][j] = dV_L_mat[k][l]
                    B_temp[i][j] = dV_R_mat[k][l]
                else:
                    '''
                    V matrices
                    '''
                    B_temp[i-1][j] = V_L_list[V_counter]
                    B_temp[i][j] = V_R_list[V_counter]
                    # increment for the next element in the list
                    V_counter += 1
            elif (i<=2 and j>1):
                '''
                all other entries are 0 matrices
                just need to get the dimensions right
                '''
                B_temp[i-1][j] = np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j]))
                B_temp[i][j] = np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j]))
                #B_temp[i-1][j] = sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
                #B_temp[i][j] = sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
                #B_temp[i-1][j] = id#'0'#sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
                #B_temp[i][j] = id#'0'#sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
            else:
                '''
                for i>=3, there will be a roll performed on the matrix, s. below.
                Therfore the j index needs to be adjusted
                '''
                # only roll every second time
                # len(block_dims_roll) depends on N
                # the larger N, the more of j-indices must be considered
                if (block_dims_roll_counter%int(len(block_dims_roll)-2) == 0):
                    block_dims_roll =  np.roll(block_dims_roll, -1)
                    #print('after roll ', block_dims_roll)

                # zero matrices
                '''
                MAYBE THERE IS A PROBLEM WITH SCALABILITY HERE WITH THE
                DIMENSION OF THE FIRST INDEX
                '''
                B_temp[i-1][j] = np.zeros(shape=(dim_after_cg, block_dims_roll[j]))
                B_temp[i][j] = np.zeros(shape=(dim_after_cg, block_dims_roll[j]))
                #print(B_temp[i-1][j].shape)
                #print(B_temp[i][j].shape)

                block_dims_roll_counter += 1

    # the entries of B_temp now must be sorted and put into a block matrix
    # e.g. for N=4 so far
    #        B_temp = ( T  0    0 )
    #                 ( 1  M_L  0 )
    #                 ( 1  M_R  0 )
    #                 ( V  M_L  0 )
    #                 ( V  M_R  0 )
    # as an array. But we want B as a matrix in the form
    #             B = ( T  0    0 )
    #                 ( 1  M_L  0 )
    #                 ( 1  M_R  0 )
    #                 ( 0  V  M_L )
    #                 ( 0  V  M_R )

    # therfore use np.roll() to bring stuff into right order
    roll_counter = 1
    for i in range(3, (2*N-3), 2):
        B_temp[i] =  np.roll(B_temp[i], roll_counter)
        B_temp[i+1] = np.roll(B_temp[i+1], roll_counter)
        roll_counter +=1


    # finally define a block matrix using hstack and vstack
    s = B_temp.shape
    s0 = s[0]
    s1 = s[1]
    B_rows = []
    #b_test = np.block(B_temp[0])
    #print(b_test)
    for i in range(s0):
        B_rows.append(np.hstack(B_temp[i]).astype(np.float64))
    #print(B_rows)

    B_matrix = np.vstack(B_rows)

    '''
    TO DO: SPARSE MATRICES
    '''

    return B_matrix

def Build_b_vector_cg(N, dim=2, chi=3):
    N = N
    d = dim
    chi = chi
    # length of b_vector
    # there are each 2 M matrices, M_R and M_L, therefore factor 2.
    # FOR N = 2:
    # Dimensions of M matrices are (d**i)*(d**i) x (d**(i+1))*(d**(i+1))
    # FOR N >= 3: we have cg. Then it is independent of N, since they all have
    # the same dimension
    # (d**chi)*(d**chi) x (d**2 * chi)*(d**2 * chi)
    # first row: T matrix has dimension 1  x  16
    # up to N=2
    b_len = 2*sum([(d**i)*(d**i) for i in range(2,3)]) + 1
    # now add the rest for N >= 3
    b_len += 2*sum([(d*chi)*(d*chi) for i in range(3,N)])

    #print('b_len', b_len)
    #print('x_vec = ', x_vec)
    #print('x_vec len = ', len(x_vec))
    #print('x_vec[0].shape[0] = ', x_vec[0].shape[0])
    # define the b vector
    b_vector = np.zeros(b_len)
    #print('b_vec = ', b_vector)
    b_vector[0] = 1
    #print('b_vec.shape = ', b_vector.shape)
    return b_vector



'''
scale it up for non iterative goarse graining
'''
def generate_linear_map_N(N=4, dim=2, chi=3, seed=0, a=-1, b=1):
#def generate_linear_map(rho, dim=2):
    # W has dimension chi x d^2, IF WE CONSIDER N=4: 2 = 4-2
    # for arbritrary N, WITHOUT iterative coarse graining
    # W has dimension chi x dim^k
    d = dim
    chi = chi
    k = N-2
    l = int(chi*(d**k))
    s = seed
    np.random.seed(s)
    # for floats in range [a,b)
    a = a
    b = b
    W = (b - a)  * np.random.random_sample((l,)) + a

    #W = np.random.randint(10, size=(l))
    W = W.reshape((chi,int(d**k)))
    #print('linear map \n', W)
    #print('rank of map', np.linalg.matrix_rank(W))
    return W

'''
PARAMETERS
seed: sets (fixed) random number
a: lower limit of range of the random number
b: upper limit of range of the random number
'''
def generate_linear_map(dim=2, chi=3, seed=0, a=-1, b=1):
#def generate_linear_map(rho, dim=2):
    # W has dimension chi x d^2
    d = dim
    chi = chi
    l = int(chi*(d**2))
    s = seed
    np.random.seed(s)
    # for floats in range [a,b)
    a = a
    b = b
    W = (b - a)  * np.random.random_sample((l,)) + a

    #W = np.random.randint(10, size=(l))
    W = W.reshape((chi,int(d**2)))
    #print('linear map \n', W)
    #print('rank of map', np.linalg.matrix_rank(W))
    return W

def generate_linear_map_2D(dim=2, chi=1, seed=0, a=-1, b=1):
    # W has dimension chi x d^4
    d = dim
    chi = chi
    l = int(chi*(d**4))
    s = seed
    np.random.seed(s)
    # for floats in range [a,b)
    a = a
    b = b
    W = (b - a)  * np.random.random_sample((l,)) + a

    #W = np.random.randint(10, size=(l))
    W = W.reshape((chi,int(d**4)))
    #print('linear map \n', W)
    #print('rank of map', np.linalg.matrix_rank(W))
    return W

'''
scale it up for non iterative goarse graining
'''
# default dimension is dim = 2 for 2 spin system
def coarse_grain_map_N(rho_N_minus_1, W, N=4, dim=2, chi=3):
    dim = dim
    #N = int(np.log(rho_N.shape[0])/np.log(dim))
    # the coarse grain map consists of a linear map W
    '''
    if we use gradient descent, the shape must be back to matrix form.
    if we dont use grad descent, it stays same anyway
    '''
    k = N-2
    W = W.reshape((chi, int(dim**k)))
    #print('W.shape', W.shape)
    # define coarse graining map
    # prepare identity matrices
    id = np.identity(dim)
    # cg
    V_L = np.kron(W, id)
    V_L_dagger = np.kron(W.conj().T, id)
    '''
    print('rho.shape', rho_N_minus_1.shape)
    print('W.shape', W.shape)
    print('V_L.shape', V_L.shape)
    print('V_L_dagger.shape', V_L_dagger.shape)
    '''
    # compute rho under a cg_map
    rho_cg_L  = V_L @ rho_N_minus_1 @ V_L_dagger

    V_R = np.kron(id, W)
    V_R_dagger = np.kron(id, W.conj().T)
    # compute rho under a cg_map
    rho_cg_R  = V_R @ rho_N_minus_1 @ V_R_dagger
    #print('rho_cg_R.shape', rho_cg_R.shape)
    #print('rho_cg_R.shape[1]', rho_cg_R.shape[1])
    #print('rho_cg_R.shape[0]', rho_cg_R.shape[0])




    return rho_cg_L, rho_cg_R

'''
NOT SCALABLE FOR NOW
'''
# default dimension is dim = 2 for 2 spin system
def coarse_grain_map(rho_N_minus_1, W, dim=2, chi=3):
    dim = dim
    #N = int(np.log(rho_N.shape[0])/np.log(dim))
    # the coarse grain map consists of a linear map W
    '''
    if we use gradient descent, the shape must be back to matrix form.
    if we dont use grad descent, it stays same anyway
    '''
    W = W.reshape((chi, int(dim**2)))
    #print('W.shape', W.shape)
    # define coarse graining map
    # prepare identity matrices
    id = sparse.identity(dim)
    # cg
    V_L = np.kron(W, id)
    V_L_dagger = np.kron(W.conj().T, id)

    #print('rho.shape', rho_N_minus_1.shape)
    #print('W.shape', W.shape)
    #print('V_L.shape', V_L.shape)
    #print('V_L_dagger.shape', V_L_dagger.shape)

    # compute rho under a cg_map
    rho_cg_L  = V_L @ rho_N_minus_1 @ V_L_dagger

    V_R = np.kron(id, W)
    V_R_dagger = np.kron(id, W.conj().T)
    # compute rho under a cg_map
    rho_cg_R  = V_R @ rho_N_minus_1 @ V_R_dagger
    #print('rho_cg_R.shape', rho_cg_R.shape)
    #print('rho_cg_R.shape[1]', rho_cg_R.shape[1])
    #print('rho_cg_R.shape[0]', rho_cg_R.shape[0])


    return rho_cg_L, rho_cg_R

'''
get the coarse graining maps V and V_dagger
default dimension is dim = 2 for 2 spin system
Here: scaleable up to N=n, WITHOUT ITERATIVE CG, but with just a simple
larger W map
'''
def cg_maps_N(W, N=4, dim=2, chi=3):
    dim = dim
    chi = chi
    k = N-2
    #N = int(np.log(rho_N.shape[0])/np.log(dim))
    # the coarse grain map consists of a linear map W
    W = W.reshape((chi, int(dim**k)))
    #W = generate_linear_map(dim=dim, chi=chi)
    #print('W.shape', W.shape)
    # define coarse graining map
    # prepare identity matrices
    id = sparse.identity(dim)
    #print('id.shape', id.shape)
    # cg
    # if we have multiple cg maps, we can use lists.
    # for now, only consider 1
    '''
    V_L = []
    V_L_dagger = []
    V_R = []
    V_R_dagger = []

    V_L.append(np.kron(W, id))
    V_L_dagger.append(np.kron(W.conj().T, id))
    V_R.append(np.kron(id, W))
    V_R_dagger.append(np.kron(id, W.conj().T))
    '''
    V_L = sparse.kron(W, id)
    V_L_dagger = sparse.kron(W.conj().T, id)
    V_R = sparse.kron(id, W)
    V_R_dagger = sparse.kron(id, W.conj().T)
    '''
    print('V_L.shape = ', V_L.shape)
    print('V_L_dagger.shape = ', V_L_dagger.shape)
    print('V_R.shape = ', V_R.shape)
    print('V_R_dagger.shape = ', V_R_dagger.shape)
    '''
    return V_L, V_L_dagger, V_R, V_R_dagger


# default dimension is dim = 2 for 2 spin system
def cg_maps(W, dim=2, chi=3):
    dim = dim
    chi = chi

    #N = int(np.log(rho_N.shape[0])/np.log(dim))
    # the coarse grain map consists of a linear map W
    W = W.reshape((chi, int(dim*dim)))
    #W = generate_linear_map(dim=dim, chi=chi)
    #print('W.shape', W.shape)
    # define coarse graining map
    # prepare identity matrices
    id = sparse.identity(dim)
    # cg
    # if we have multiple cg maps, we can use lists.
    # for now, only consider 1
    '''
    V_L = []
    V_L_dagger = []
    V_R = []
    V_R_dagger = []

    V_L.append(np.kron(W, id))
    V_L_dagger.append(np.kron(W.conj().T, id))
    V_R.append(np.kron(id, W))
    V_R_dagger.append(np.kron(id, W.conj().T))
    '''
    V_L = sparse.kron(W, id)
    V_L_dagger = sparse.kron(W.conj().T, id)
    V_R = sparse.kron(id, W)
    V_R_dagger = sparse.kron(id, W.conj().T)


    return V_L, V_L_dagger, V_R, V_R_dagger

#
