import numpy as np
import scipy.sparse as sparse
from scipy.linalg import block_diag

import define_system as ds
import define_SDP as dsdp
import utility as ut


'''
get output of the Hamiltonian
'''

def testing_ptrace():
    ma = np.arange(1,17,1)
    ma =  ma.reshape((4,4))
    mat = np.arange(0,16,1)
    mat =  mat.reshape((4,4))
    mat = ma #+ 1j*mat
    #print(mat)
    #print(mat.conj().T)
    mat2 = mat.reshape((2,2,2,2))
    #print(mat2)
    #print(mat2.conj().T)

    id  = np.array([[1,0], [0,1]])
    id3 = np.identity(3)
    sigma_x = np.array([[0,1], [1,0]])

    A = np.kron(id,sigma_x)
    print("A= \n", A)
    B = np.kron(id3,sigma_x)
    print("B= \n", B)
    ptr1 = ut.ptrace(A, (2,2), 1)
    ptr2 = ut.ptrace(A,(2,2), 2)

    ptr1_B = ut.ptrace(B, (3,2), 1)
    ptr2_B = ut.ptrace(B,(3,2), 2)

    #print(mat)
    print(ptr1)
    print(ptr2)

    print(ptr1_B)
    print(ptr2_B)
    return 0

def testing_Build_H_TFIM_Interaction_Term():
    interaction_term = ds.Build_H_TFIM_Interaction_Term()
    return print(interaction_term)

#testing_Build_Hamiltonian_Interaction_Term()


# create some random rho matrices
# basically some random complex 2^N x 2^N matrix
def rhos_example(N):
    np.random.seed(1)
    rho = []
    #include the N-th index
    for i in range(2, N+1):
        rho.append(np.random.randn(2**i,2**i)) #+ 1j*np.random.randn(2**i,2**i))
    return rho
# create an example of rhos
rho_example_list = rhos_example(4)
rho2 = rho_example_list[0]
print('rho2', rho2)
print('tr rho2', np.trace(rho2))

'''
N is dimension of system
'''
def testing_decompose_matrix(N):
    print('dimension of the system: ', N)

    rho_example_list = rhos_example(N)
    #print('rho2 = ', rho_list[0])
    #print('rho3 = ', rho_list[1])
    #print('rho4 = ', rho_list[2])
    #print('==========================')
    elements = []
    basis = []
    for i in range(int(len(rho_example_list))):
        ele, base = dsdp.decompose_matrix(rho_example_list[i])
        elements.append(ele)
        basis.append(base)

    return elements, basis

#elements, basis = testing_decompose_matrix(2)
#print('elements of the matrix', elements)
#print('basis fo the matrix', basis)

# testing how the scipy block_diag() function works
def testing_block_diag():
    N = 1
    np.random.seed(1)
    A = np.random.randn(2**N, 2**N)# + 1j*np.random.randn(2**N,2**N)

    np.random.seed(2)
    B = np.random.randn(2, 2)# + 1j*np.random.randn(2**N,2**N)
    np.random.seed(3)
    C = np.random.randn(2, 2)
    print('A = ', A)
    print('B = ', B)
    print('C = ', C)
    A_B = block_diag(A,B)
    AB = zip(A, B)
    Z = list(block_diag(AB))
    D = block_diag(A_B, C)
    E = list(map(block_diag, *zip((A,B,C))))
    return D, E, Z
#D, E, Z = testing_block_diag()

#print('D block_diag = ', D)
#print('E block_diag = ', E[0])
#print('Z block_diag = ', Z)

def testing_Build_x_vector(rho_list):
    x_vec = dsdp.Build_x_vector(rho_list)

    return x_vec
# print the list of rhos that was generated above
#print('rho_example_list', rho_example_list)
x_vec = testing_Build_x_vector(rho_example_list)
#print('x_vec ', x_vec)

def testing_Build_basis_vector(rho_list):
    basis_vec = dsdp.Build_basis_vector(rho_list)
    return basis_vec
basis_vec = dsdp.Build_basis_vector(rho_example_list)
#print(basis_vec)
#print(len(basis_vec))

def testing_set_basis_vector(N):
    basis_vec = dsdp.set_basis_vector(N)
    return basis_vec
basis_vector = testing_set_basis_vector(4)
print('basis_vector', basis_vector)
print(len(basis_vector))
'''
print('basis', basis_vec[0][0] == basis_vector[0][0])
print('basis1', basis_vec[1][5] == basis_vector[1][5])
print('basis2', basis_vec[2][42] == basis_vector[2][42])
'''
def testing_Build_A_matrices_old(rho_list):
    A = dsdp.Build_A_matrices_old(rho_list)
    return A
A_matrices = testing_Build_A_matrices_old(rhos_example(4))
#print(A_matrices[34])
#print(A_matrices[0].shape)

def testing_Build_A_matrices(N, basis_vec):
    A = dsdp.Build_A_matrices(N, basis_vec)
    return A
#print('basis vector', basis_vector)
#print(len(basis_vector))

A_mat = testing_Build_A_matrices(4, basis_vector)
#print(A_mat[34] == A_matrices[34])



def testing_Build_M_matrices(basis_vec):
    M_R_list, M_L_list = dsdp.Build_M_matrices(basis_vec)
    return M_R_list, M_L_list

basis_vector = testing_set_basis_vector(4)
M_R_list, M_L_list = testing_Build_M_matrices(basis_vec)
'''
print('mr= ', M_R_list)
print('ml= ', M_L_list)
print('len mr= ', len(M_R_list))
print('len ml= ', len(M_L_list))
'''
def testing_B_matrix_and_b_vector(M_R_list, M_L_list):
    B_matrix, b_vector = dsdp.Build_B_matrix_and_b_vector(M_R_list, M_L_list)
    return B_matrix, b_vector

B_matrix, b_vector = testing_B_matrix_and_b_vector(M_R_list, M_L_list)
B_mat = B_matrix[0][0:16]
print('B_matrix.shape', B_matrix.shape)
print('B_mat', B_matrix[0][0:16])
rho2_flatten = rho2.flatten()
# basically trace
print('B_mat @ rho2', B_mat @ rho2_flatten)
B_first_row = B_matrix[0]
print('B_first_row', B_first_row)
print('B_first_row.shape', B_first_row.shape)

#print('b_vec ', b_vector)
#print('B.shape', B_matrix.shape)

#print('mr', M_R_list[1].shape)
#print('ml', M_L_list[1].shape)
#print('mr2', M_R_list[2].shape)
#print('ml2', M_L_list[2].shape)


#print(B_matrix)
#print(B_matrix.shape)

def testing_Build_c_vector(ham, basis_vec, N):
    c_vec = dsdp.Build_c_vector(ham, basis_vec, N)
    return c_vec


ham = ds.Build_H_TFIM_Interaction_Term()
c_vec = testing_Build_c_vector(ham, basis_vec, 4)
print('c = ', len(c_vec))
print('c', c_vec)


#print(res)
#print(res.shape)
x_vec = np.block(x_vec)
#print(c_vec @ x_vec)

# create an example of rhos
rho_example_list = rhos_example(4)
rho3 = rho_example_list[1]
rho4 = rho_example_list[2]
print('rho3 \n', rho3)

'''
def testing_generate_linear_map(rho, dim):
    W = dsdp.generate_linear_map(rho, dim=dim)
    return W
W = testing_generate_linear_map(rho3, dim=2)
'''
'''
def testing_coarse_grain_map_L(rho, dim):
    rho_cg_L = dsdp.coarse_grain_map_L(rho, dim=dim)
    return rho_cg_L
rho_cg_L = testing_coarse_grain_map_L(rho3, dim=2)

def testing_coarse_grain_map_R(rho, dim):
    rho_cg_R = dsdp.coarse_grain_map_R(rho, dim=dim)
    return rho_cg_R

rho_cg_R = testing_coarse_grain_map_R(rho3, dim=2)
print('rho_cg_R \n', rho_cg_R)
print('rho_cg_R.shape', rho_cg_R.shape)
'''

'''
def testing_coarse_grain_map(rho_N_minus_1, rho_N, dim=2):
    rho_cg_L, rho_cg_R, omega = dsdp.coarse_grain_map(rho_N_minus_1, rho_N, dim=dim)
    return rho_cg_L, rho_cg_R, omega

# omega
rho_cg_L, rho_cg_R, omega = testing_coarse_grain_map(rho3, rho4, dim=2)
print('rho_cg_L \n', rho_cg_L)
print('rho_cg_L.shape = ', rho_cg_L.shape)

print('rho_cg_R \n', rho_cg_R)
print('rho_cg_R.shape = ', rho_cg_R.shape)

print('omega \n', omega)
print('omega.shape = ', omega.shape)

ptr_R = ut.ptrace(omega, (4, 2), 2)
print(ptr_R.shape)
print(ptr_R)

ptr_L = ut.ptrace(omega, (2, 4), 1)
print(ptr_L.shape)
print(ptr_L)
'''
