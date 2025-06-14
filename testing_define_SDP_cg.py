import numpy as np
import define_SDP_cg as dsdp_cg
import utility as ut
import define_system as ds

def testing_Build_x_vector(N, dim=2, chi=3):
    x_vec = dsdp_cg.Build_x_vector_cp_cg(N, dim=dim, chi=chi)

    return x_vec

'''
x_vec = testing_Build_x_vector(5, dim=2, chi=3)
print('x_vec = ', x_vec)
print('x_vec.shape = ', x_vec.shape)
'''

'''
test the basis vector function
'''
def testing_set_basis_vector_cg(n_cols, n_rows):
    basis_vec = dsdp_cg.set_basis_vector_cg(n_cols, n_rows)
    return basis_vec

chi = 3
dim = 2
n_cols_in = dim**3
n_rows_in = dim**3
basis_vec_in = testing_set_basis_vector_cg(n_cols_in, n_rows_in)

n_cols_out = chi*dim
n_rows_out = chi*dim
basis_vec_out = testing_set_basis_vector_cg(n_cols_out, n_rows_out)

#print('basis_vec', basis_vec)
#print('len(basis_vec)', len(basis_vec))
#print('basis_vec[0]', basis_vec[0])


def testing_get_basis_vector_cg(N, dim=2, chi=3):
    basis_vector = dsdp_cg.get_basis_vector_cg(N, dim=2, chi=3)
    return basis_vector
'''
b = testing_get_basis_vector_cg(4, dim=2, chi=3)
print('b', b)
print('b[0]', b[0])
print('len b', len(b))
'''


'''
test the matrix rep function
'''
def testing_Build_matrix_rep(lin_op, basis_in, basis_out, *args):
    m = ut.Build_matrix_rep(lin_op, basis_in, basis_out, *args)
    return m

W = dsdp_cg.generate_linear_map(dim=dim, chi=chi, seed=5)
lin_op = ut.linear_op_cg_maps
V_mat_L = testing_Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'L',
                                dim, chi, W)
V_mat_R = testing_Build_matrix_rep(lin_op, basis_vec_in, basis_vec_out, 'R',
                                dim, chi, W)
#print('V_mat_L', V_mat_L)
print('V_mat_L.shape', V_mat_L.shape)
print('V_mat_R.shape', V_mat_R.shape)

'''
test, if the V is indeed the right matrix, by preparing some random rho and
compare

can do it better without optional arguments
'''
def check_V_matrix(V_mat, *args):
    rho = np.random.randint(10, size=(8, 8))
    rho_flatten= rho.flatten()
    V_L, V_L_dagger, V_R, V_R_dagger = dsdp_cg.cg_maps()

    for arg in args:
        if (arg == 'L'):
            V_compare_L = V_L @ rho @ V_L_dagger
            V_mat_L = V_mat @ rho_flatten
            V_mat_L = V_mat_L.reshape((6,6))
            print(V_compare_L)
            print(V_mat_L)
            return print(V_mat_L == V_compare_L)
        if (arg == 'R'):
            V_compare_R = V_R @ rho @ V_R_dagger
            V_mat_R = V_mat @ rho_flatten
            V_mat_R = V_mat_R.reshape((6,6))

            print(V_compare_R)
            print(V_mat_R)
            return print(V_mat_R == V_compare_R)

    return print('No argument for L or R')

'''
l = check_V_matrix(V_mat_L, 'L')
r = check_V_matrix(V_mat_R, 'R')
'''

'''
test, if the M matrices representing the ptr(omega_4) works as intended
'''
def check_M_matrix(M_mat, dim=2, chi=3, L=False, R=False):
    d = dim
    chi = chi
    # set some random omega with right dimension
    omega = np.random.randint(10, size=(d**2 * chi, d**2 * chi))
    omega_flatten = omega.flatten()
    ptr_R = ut.ptrace(omega, (d*chi, d), 2)
    ptr_L = ut.ptrace(omega, (d, d*chi), 1)

    if (L==True):
        M_L_times_omega = M_mat @ omega_flatten
        M_L_times_omega = M_L_times_omega.reshape((d*chi, d*chi))
        print('ptr_L',  ptr_L)
        print('M_L_times_omega', M_L_times_omega)
        return  print(M_L_times_omega == ptr_L)
    if (R==True):
        M_R_times_omega = M_mat @ omega_flatten
        M_R_times_omega = M_R_times_omega.reshape((d*chi, d*chi))
        print('ptr_R',  ptr_R)
        print('M_R_times_omega', M_R_times_omega)
        return  print(M_R_times_omega == ptr_R)
    return print('L and R are both false')
'''
chi = 3
dim = 2
n_cols_in = dim**2 * chi
n_rows_in = dim**2 * chi
M_basis_vec_in = testing_set_basis_vector_cg(n_cols_in, n_rows_in)

# we trace out one dimension, so output is input_dim/ 1dim
n_cols_out = dim*chi
n_rows_out = dim*chi
M_basis_vec_out = testing_set_basis_vector_cg(n_cols_out, n_rows_out)

lin_op = ut.linear_op_ptr
M_R = testing_Build_matrix_rep(lin_op, M_basis_vec_in, M_basis_vec_out, (dim*chi, dim), 2)
M_L = testing_Build_matrix_rep(lin_op, M_basis_vec_in, M_basis_vec_out, (dim, dim*chi), 1)
print('M_R.shape', M_R.shape)
l = check_M_matrix(M_L, L=True)
r = check_M_matrix(M_R, R=True)
'''

'''
test the B matrix
'''
def testing_B_matrix_cg(N, W, dim=2, chi=3, gradient=False):
    B_matrix = dsdp_cg.Build_B_matrix_cg(N, W, dim=dim, chi=chi)
    grad_B = dsdp_cg.Build_B_matrix_cg(N, W, dim=dim, chi=chi, gradient=True)
    return B_matrix, grad_B
'''
dim = 2
chi = 3
W = dsdp_cg.generate_linear_map(dim=dim, chi=chi)
N=4
B_matrix, grad_B = testing_B_matrix_cg(N, W, dim=2, chi=3)
B_mat = B_matrix[0][0:16]
print('B_matrix.shape', B_matrix.shape)
print('B_mat', B_matrix[0][0:16])
#rho2_flatten = rho2.flatten()
# basically trace
#print('B_mat @ rho2', B_mat @ rho2_flatten)
B_first_row = B_matrix[0]
print('B_first_row', B_first_row)
print('B_first_row.shape', B_first_row.shape)

print('grad_B', grad_B)
print('grad_B.shape', grad_B.shape)
print('grad_B.flatten', grad_B.flatten())
for i in range(0, len(grad_B.flatten())):
    print(grad_B.flatten()[i])
'''

'''
check if dB is actually calculated correctly
MAYBE THERE IS SOME MISTAKE BC OF THE W_DAGGER
'''
def check_dB():
    N=4
    # get a small cg map
    d = dim = 2
    chi = 3
    W = dsdp_cg.generate_linear_map(dim=dim, chi=chi, seed=0)
    print('W', W)
    V_L_mat, V_R_mat = dsdp_cg.get_V_matrices(N, W, dim=dim, chi=chi)
    dV_L, dV_R, dV_L_mat, dV_R_mat = ut.get_derivative_V(N, W, dim=dim, chi=chi)
    # dV_L and dV_R have arrays as element; get them out
    dV_L = dV_L[0]
    dV_R = dV_R[0]

    # take a look at the derivatives [at point (i,j)]
    #print('dV_L_mat = ', dV_L_mat)
    #print('dV_R_mat = ', dV_R_mat)
    #V_L_list, V_R_list = ut.get_derivative_V(N, W, dim=dim, chi=chi)

    '''
    get V and its derivative manually
    '''
    n_cols_in = d**3
    n_rows_in = d**3
    basis_vec_in = dsdp_cg.set_basis_vector_cg(n_cols_in, n_rows_in)

    n_cols_out = chi*d
    n_rows_out = chi*d
    basis_vec_out = dsdp_cg.set_basis_vector_cg(n_cols_out, n_rows_out)

    n_rows = np.prod(basis_vec_in[0].shape)
    n_cols = np.prod(basis_vec_out[0].shape)
    # initialize matrix
    mat_L = np.zeros((n_cols, n_rows))
    mat_R = np.zeros((n_cols, n_rows))

    mat_L_list = []
    mat_R_list = []

    # initialize variable for derivative
    dV_dW_L = 0
    dV_dW_R = 0
    # get the E_LR (dagger) and V_LR (daggger) for the derivative
    V_L, V_L_dagger, V_R, V_R_dagger = dsdp_cg.cg_maps(W, dim=dim, chi=chi)
    #print('V_L', V_L)
    #print('V_L_dagger', V_L_dagger)
    # create index tuples for initializing the E matrices
    k = W.shape[0]
    l = W.shape[1]

    indices = []
    # create all possible indices of the W matrix
    for i in range(0, k):
        for j in range(0, l):
            indices.append((i,j))

    for index_tuple in indices:
        #print('############################################################')
        #print('INDEX_TUPLE', index_tuple)
        E_L, E_L_dagger, E_R, E_R_dagger = dsdp_cg.get_E_matrices(N, index_tuple, dim=dim, chi=chi)
        for i in range(0, n_cols):
            for j in range(0, n_rows):
                # use product rule
                mat_L[i][j] = (np.trace((E_L @ basis_vec_in[j] @ V_L_dagger).conj().T @ basis_vec_out[i]) + np.trace((V_L @ basis_vec_in[j] @ E_L_dagger).conj().T @ basis_vec_out[i]))

                mat_R[i][j] = (np.trace((E_R @ basis_vec_in[j] @ V_R_dagger).conj().T @ basis_vec_out[i]) + np.trace((V_R @ basis_vec_in[j] @ E_R_dagger).conj().T @ basis_vec_out[i]))
        #print(f'dV_L matrix at index {index_tuple}\n', mat_L)
        dV_dW_L += mat_L
        dV_dW_R += mat_R
        # somehow this doesnt work
        #mat_L_list.append(mat_L)
        #mat_R_list.append(mat_R)
    '''
    IRGENDWAS WIRD UEBERSCHRIEBEN. mat_L_list[0] = mat_L_list[1] = mat_L_list[2]
    WIESO AUCH IMMER
    '''
    # sum over all indices
    #mat_L_sum = sum(mat_L_list)
    #mat_R_sum = sum(mat_R_list)
    #V_L_sum = sum(V_L_list)
    #V_R_sum = sum(V_R_list)


    print(np.linalg.norm(dV_dW_L-dV_L))
    print(np.linalg.norm(dV_dW_R-dV_R))
    '''
    print('dV_dW_L ', dV_dW_L)
    print('dV_dW_L[0] ', dV_dW_L[0])
    print('dV_dW_L[0][0] ', dV_dW_L[0][0])
    print('dV_L ', dV_L)
    print('dV_L[0] ', dV_L[0])
    print('dV_L[0][0] ', dV_L[0][0])
    for i in range(n_cols):
        for j in range(n_rows):
            print(f'dV_dW_L[{i}][{j}] == dV_L[{i}][{j}]', dV_dW_L[i][j] == dV_L[i][j])
            print(f'dV_dW_R[{i}][{j}] == dV_R[{i}][{j}]', dV_dW_R[i][j] == dV_R[i][j])
    '''
    B_mat = dsdp_cg.Build_B_matrix_cg(N, W, dim=dim, chi=chi, gradient=False,
                                        grad_kl=False, index_tuple=(0,0))
    print('B_mat.shape', B_mat.shape)
    grad_B = dsdp_cg.Build_B_matrix_cg(N, W, dim=dim, chi=chi, gradient=True,
                                        grad_kl=False, index_tuple=(0,0))
    print('grad_B.shape', grad_B.shape)
    return grad_B
    #return V_L, V_R, dV_L, dV_R
'''
grad_B = check_dB()
m, n = grad_B.shape

print('shape of dB', grad_B.shape)
'''

'''
for i in range(m):
    for j in range(n):
        print(f'grad_B[{i}][{j}] = ', grad_B[i][j])
'''
'''
V_L, V_R, dV_L, dV_R = check_dB()
print('V_L\n', V_L[0])
print('V_L.shape\n', V_L[0].shape)
print('V_R\n', V_R[0])
print('dV_L\n', dV_L[0])
print('dV_R\n', dV_R[0])
print('dV_R.shape\n', dV_R[0].shape)
'''

def testing_get_energy_and_dervative_of_energy(W, N, ham, dim=2, chi=3):
    e, de_dW = dsdp_cg.get_energy_and_derivative_of_energy(W, N, ham, dim=dim, chi=chi)
    return e, de_dW
'''
N=4
W = dsdp_cg.generate_linear_map(dim=2, chi=3, seed=0)
ham = ds.Build_H_TFIM_Interaction_Term()
e, de_dW = testing_get_energy_and_dervative_of_energy(W, N, ham, dim=2, chi=3)
print('energy = ', e)
print('de_dW', de_dW)
'''

def testing_check_SDP_grad_formula(N, W, ham, dim=2, chi=3):
    placeholder = dsdp_cg.check_SDP_grad_formula(N, W, ham, dim=2, chi=3)
    return placeholder
'''
N = 4
ham = ds.Build_H_TFIM_Interaction_Term()
W = dsdp_cg.generate_linear_map(dim=2, chi=3, seed=0)
a = testing_check_SDP_grad_formula(N, W, ham, dim=2, chi=3)
'''
def testing_check_grad_function():
    N = 4
    Delta = 1
    ham = ds.Build_H_XXZ(Delta=Delta)
    a = -1
    b = 1
    seed = 0
    W = dsdp_cg.generate_linear_map_N(N=N, dim=2, chi=3, seed=seed, a=a, b=b)
    for s in range(10):
        result = dsdp_cg.check_grad_function(W, N, ham, s)
        print('result', result)
    return 0
#testing_check_grad_function()

'''
for s in range(6,8):
    W = dsdp_cg.generate_linear_map(dim=2, chi=3, seed=s)
    a = testing_check_SDP_grad_formula(N, W, ham, dim=2, chi=3)
'''


'''
stuff for testing the coarse graining map, maybe do it prettier later
'''
'''
def testing_rhos_cp_variable(N, dim=2):
    rhos = ssdp.rhos_cp_variable(N, dim=dim)
    return rhos
N = 10
dim = 2
chi = 3
rhos = testing_rhos_cp_variable(N)
rho3 = rhos[1]
rho4 = rhos[2]
rho_N = rhos[N-3]
W = dsdp_cg.generate_linear_map_N(N=N, dim=dim, chi=chi, seed=5)
rho_cg_L, rho_cg_R, omega = dsdp_cg.coarse_grain_map_N(rho_N, W, N=N, dim=dim,
                                                    chi=chi)
print('rho_cg_L.shape = ', rho_cg_L.shape)
print('rho_cg_R.shape = ', rho_cg_R.shape)
print('omega.shape = ', omega.shape)
'''




#
