"""
This module contains functions (and classes) that help define our semi-definite problem
"""

import numpy as np
from numpy import linalg as LA
import cvxpy as cp
import scipy.sparse as sparse
from scipy.linalg import block_diag

import utility as ut


def decompose_matrix(rho):
    # get shape of matrix in both dimensions
    rho_shape_0 = rho.shape[0]
    rho_shape_1 = rho.shape[1]

    basis = []

    # create a set of orthonormal basis e.g. in 2D
    # {[[1,0], [0,0]],[[0,1], [0,0]],  [[0,0], [1,0]], [[0,0], [0,1]]}
    # and so on for higher dimensions
    for i in range(0, rho_shape_0):
        for j in range(0, rho_shape_1):
            # create a matrix with right dimensions on only 0 as entries
            temp_matrix = np.zeros((rho_shape_0, rho_shape_1))
            # fill the i,j-th entry with a single 1
            temp_matrix[i][j] = 1
            # sparse it
            """
            Comment out to get a better view vor testing
            """
            # temp_matrix = sparse.csr_matrix(temp_matrix)
            basis.append(temp_matrix)

    # get the individual matrix elements as a flattened array
    elements = rho.flatten()

    return elements, basis


# needed for testing, if everything works
def Build_x_vector(rho_list):
    x_vec = []
    # basis_vec = []

    # get the matrix elements
    for i in range(0, len(rho_list)):
        elements, basis = decompose_matrix(rho_list[i])
        # x_vec now has whole arrays as entries
        x_vec.append(elements)

    return x_vec


# needed for testing, get values out of a cp variable
def Build_x_vector2(rho_list):
    x_vec = []
    # basis_vec = []

    # get the matrix elements
    for i in range(0, len(rho_list)):
        elements, basis = decompose_matrix(rho_list[i].value)
        # x_vec now has whole arrays as entries
        x_vec.append(elements)
    x_vec = np.block(x_vec)
    return x_vec


# get x vector as cp Variable
def Build_x_vector_cp(N):
    # get length of the vector
    l = sum([(2**i) * (2**i) for i in range(2, N + 1)])
    x_vec_cp = cp.Variable((l, 1))

    return x_vec_cp


# used for testing, if everything works
def Build_basis_vector(rho_list):
    basis_vec = []

    # get the basis
    for i in range(0, len(rho_list)):
        elements, basis = decompose_matrix(rho_list[i])
        # basis_vec now has whole arrays as entries
        basis_vec.append(basis)

    return basis_vec


# construct the basis needed in the problem
def set_basis_vector(N):
    # for the dimension of the basis
    l = [(2**i) for i in range(2, N + 1)]
    # want o get the basis vec as 2D object
    basis = []
    basis_vec = []

    # create a set of orthonormal basis e.g. in 2D
    # {[[1,0], [0,0]],[[0,1], [0,0]],  [[0,0], [1,0]], [[0,0], [0,1]]}
    # and so on for higher dimensions
    for i in range(len(l)):
        for j in range(0, l[i]):
            for k in range(0, l[i]):
                # create a matrix with right dimensions on only 0 as entries
                temp_matrix = np.zeros((l[i], l[i]))
                # fill the i,j-th entry with a single 1
                temp_matrix[j][k] = 1
                # sparse it
                """
                Comment out to get a better view vor testing
                """
                # temp_matrix = sparse.csr_matrix(temp_matrix)
                basis.append(temp_matrix)
        basis_vec.append(basis)
        # reset basis
        basis = []

    return basis_vec


"""
FOR TESTING
build all A matrices using block_diag from scipy
(which basically is the direct sum)
"""


def Build_A_matrices_old(rho_list):
    A_vec = []
    basis_vec = []
    # need the dimension of the A matrix
    A_dims = []
    # get the matrix elements
    for i in range(0, len(rho_list)):
        elements, basis = decompose_matrix(rho_list[i])
        basis_vec.append(basis)

        # keep track of the dimension of all the rho matrices
        A_dims.append(len(basis))
    # print('A dims ', A_dims)
    basis_vec = tuple(basis_vec)
    # print('basis_vec = ', basis_vec)

    # A matrix is the form
    # A^1_i = [[E^1_i, 0, 0], [0, 0, 0], [0, 0, 0]]
    # A^2_i = [[0, 0, 0], [0, E^2_i, 0], [0, 0, 0]]
    # etc... in the appropiate dimension

    # depending on how many different sizes of basis we have
    # (e.g. 4x4, 16x16, 64x64 are three basis with different size)
    #
    zero_matrices = []
    for j in range(0, len(A_dims)):
        # print(j)
        # create all neccessary zero matrices that are used in the block matrix
        zero_matrix = np.zeros((int(np.sqrt(A_dims[j])), int(np.sqrt(A_dims[j]))))
        zero_matrices.append(zero_matrix)
    # print('zero', zero_matrices)

    # A_mat = zero_matrices
    # print('A_mat = ', A_mat)
    for k in range(0, len(A_dims)):
        # print('k ', k)
        # print('basis_vec[k] = ', basis_vec[k])
        for l in range(A_dims[k]):
            # print('l ', l)
            # print('basis_vec[k][l] = ', basis_vec[k][l])
            # print('ind ', ind)

            # replace a zero matrix with an actual basis, so this list consists
            # of the diagonal components of the desired block matrix
            A_mat = zero_matrices.copy()

            # print(basis_vec[k][l])
            A_mat[k] = basis_vec[k][l]
            # print('A_mat[k] = ', A_mat[k])
            # print('A-mat', A_mat)

            A_vec.append(A_mat)

    # print(A_vec[0])
    # print(len(A_vec))
    # get the block matrices
    A_matrices_final = []
    for i in range(len(A_vec)):
        res = list(map(block_diag, *zip(tuple(A_vec[i]))))[0]
        A_matrices_final.append(res)

        """
        IMPLEMENT SPARSE MATRIX LATER
        """
    return A_matrices_final


"""
build all A matrices using block_diag from scipy for testing
(which basically is the direct sum)
"""


def Build_A_matrices(N, basis_vec):
    A_vec = []
    # need the dimension of the A matrix
    A_dims = [(2**i) * (2**i) for i in range(2, N + 1)]
    # print('A_dims_new', A_dims)

    basis_vec = tuple(basis_vec)
    # print('basis_vec = ', basis_vec)

    # A matrix is the form
    # A^1_i = [[E^1_i, 0, 0], [0, 0, 0], [0, 0, 0]]
    # A^2_i = [[0, 0, 0], [0, E^2_i, 0], [0, 0, 0]]
    # etc... in the appropiate dimension

    # depending on how many different sizes of basis we have
    # (e.g. 4x4, 16x16, 64x64 are three basis with different size)
    #
    zero_matrices = []
    for j in range(0, len(A_dims)):
        # print(j)
        # create all neccessary zero matrices that are used in the block matrix
        zero_matrix = np.zeros((int(np.sqrt(A_dims[j])), int(np.sqrt(A_dims[j]))))
        zero_matrices.append(zero_matrix)
    # print('zero', zero_matrices)

    # A_mat = zero_matrices
    # print('A_mat = ', A_mat)
    for k in range(0, len(A_dims)):
        # print('k ', k)
        # print('basis_vec[k] = ', basis_vec[k])
        for l in range(A_dims[k]):
            # print('l ', l)
            # print('basis_vec[k][l] = ', basis_vec[k][l])
            # print('ind ', ind)

            # replace a zero matrix with an actual basis, so this list consists
            # of the diagonal components of the desired block matrix
            A_mat = zero_matrices.copy()

            # print(basis_vec[k][l])
            A_mat[k] = basis_vec[k][l]
            # print('A_mat[k] = ', A_mat[k])
            # print('A-mat', A_mat)

            A_vec.append(A_mat)

    # print(A_vec[0])
    # print(len(A_vec))
    # get the block matrices
    A_matrices_final = []
    for i in range(len(A_vec)):
        res = list(map(block_diag, *zip(tuple(A_vec[i]))))[0]
        A_matrices_final.append(res)

        # IMPLEMENT SPARSE MATRIX LATER

    return A_matrices_final


# build the M matrix
# return both Tr_R and Tr_L


# N should be a variable later in a main function
# set to N=4 in the meantime
def Build_M_matrices(basis_vec):
    # get the dimension of the orthonormal matrices, to create the M matrix
    # !!!!!!!!!!!!!!!!!!!!
    # maybe absoulte useless, if the dimensions can also calculated from N directly
    # !!!!!!!!!!!!!!!!!!!!!!
    dim_E = []
    for i in range(len(basis_vec)):
        # from each set of basis [first index] (bc the sets have different dimensions),
        # consider the first element [second index]
        dim_E.append(basis_vec[i][0].shape[0])
    # 4, 8, 16
    # print('dim E', dim_E)

    # create list for all the M
    M_R_list = []
    M_L_list = []

    # initilaize the M matrices
    # since m in {3,..., N}, we need each N-2 M_R and M_L
    # dim_E is a list of length N-1
    N = len(dim_E) + 1
    # print('N = ', N)
    for k in range(0, N - 2):
        # print('k= ', k)
        # create an empty array to store the matrix elements
        # (for the most general datatype, since we want to store some cvx matrix calculations)
        M_R = np.empty(
            shape=(dim_E[k] * dim_E[k], dim_E[k + 1] * dim_E[k + 1]), dtype="object"
        )
        M_L = np.empty(
            shape=(dim_E[k] * dim_E[k], dim_E[k + 1] * dim_E[k + 1]), dtype="object"
        )
        # print('=================')
        # print(dim_E[k+1]*dim_E[k+1])
        # print(dim_E[k]*dim_E[k])
        for j in range(0, dim_E[k + 1] * dim_E[k + 1]):
            # print('j= ', j)
            # at least the dimensions are right
            for i in range(0, dim_E[k] * dim_E[k]):
                # print('i= ', i)
                # print(basis_vec[k+1][j])

                # since ptr acts on a basis, we dont need cp partial trace, np stuff is fine
                # hermitian transpose in done via .conj().T
                # the .H operator only works on matrix objects, not on arrays

                M_R[i][j] = np.trace(
                    (ut.ptrace(basis_vec[k + 1][j], (dim_E[k], 2), 2)).conj().T
                    @ basis_vec[k][i]
                )
                M_L[i][j] = np.trace(
                    (ut.ptrace(basis_vec[k + 1][j], (2, dim_E[k]), 1)).conj().T
                    @ basis_vec[k][i]
                )
                # print('i= ', i)
        # basis_vec is already a sparse matrix, so the entries of the M matrix should also be sparse
        M_R_list.append(M_R)
        M_L_list.append(M_L)

    return M_R_list, M_L_list


# given are the parameters
# x_vectors and the M matrices in M_R_list/M_L_list
def Build_B_matrix_and_b_vector(M_R_list, M_L_list):

    # build the matrix that represents the trace
    def Build_T_matrix():
        T = np.zeros(shape=(1, 16))
        for i in range(0, 16):
            if i % 5 == 0:
                T[0][i] = 1
        return T

    T = Build_T_matrix()

    # get N
    N = len(M_R_list) + 2
    # print('N = ', N)
    # print('mr', M_R_list[1].shape)
    # print('ml', M_L_list[1].shape)

    # length of b_vector
    # there are each 2 M matrices, M_R and M_L, therefore factor 2.
    # Dimensions of M matrices are (2**i)*(2**i) x (2**(i+1))*(2**(i+1))
    # first row: T matrix has dimension 1  x  16
    b_len = 2 * sum([(2**i) * (2**i) for i in range(2, N)]) + 1
    # print('b_len', b_len)
    # print('x_vec = ', x_vec)
    # print('x_vec len = ', len(x_vec))
    # print('x_vec[0].shape[0] = ', x_vec[0].shape[0])
    # define the b vector
    b_vector = np.zeros(b_len)
    # print('b_vec = ', b_vector)
    b_vector[0] = 1
    # print('b_vec.shape = ', b_vector.shape)
    """
    # get the n dimensions of the mxn blockmatrices, depending on N
    # N as global variable???
    """

    block_dims = [(2**i) * (2**i) for i in range(2, N + 1)]
    # print(block_dims)
    # copy the dimensions, to get the indeces in an easier way
    # for the zeros matrices
    block_dims_roll = block_dims.copy()
    # print(block_dims_roll)

    # og plan: use np.block and add blocks

    # B is going to be a block matrix
    # It contains of (2N-3)x(N-1) blocks
    # initialize an empty array of that size
    B_temp = np.empty(shape=(2 * N - 3, N - 1), dtype="object")

    # counter to track, how many times the first for loop is running
    # can be used as index to get the right dimension out of the block_dims
    dim_counter = 0
    # as well to track the indices for the M_lists
    M_counter = 0
    # for loop i index: increment by 2
    # e.g. we get for N=4: i=0,2,4
    # so we can fill out i=0, then i=1,2 togehter and then i=3,4 components
    block_dims_roll_counter = 0

    for i in range(0, (2 * N - 3), 2):
        # print('i', i)
        for j in range(N - 1):
            # print('j1', j)
            # first row of B
            # 1x16, 1x64, 1x256,.... etc matrix
            if i == 0 and j == 0:
                # build the trace matrix T (1x16)
                B_temp[i][j] = T
                # this was a placeholder
                # B_temp[i][j] = np.zeros(shape=(1, block_dims[dim_counter]))
                # B_temp[i][j] = sparse.csr_matrix(np.zeros(shape=(1, block_dims[dim_counter])))
                # B_temp[i][j] = id#'T'#sparse.csr_matrix(np.zeros(shape=(1, block_dims[dim_counter])))
            elif i == 0 and j != 0:
                # block_dims has length N-1, so just use the j index to read out that list
                B_temp[i][j] = np.zeros(shape=(1, block_dims[j]))
                # B_temp[i][j] = sparse.csr_matrix(np.zeros(shape=(1, block_dims[j])))
                # B_temp[i][j] = id#'0'#sparse.csr_matrix(np.zeros(shape=(1, block_dims[j])))
            elif i != 0 and j == 0:
                # identity matrices
                B_temp[i - 1][j] = np.identity(block_dims[dim_counter])
                # B_temp[i-1][j] = sparse.csr_matrix(np.identity(block_dims[dim_counter]))
                B_temp[i][j] = np.identity(block_dims[dim_counter])
                # B_temp[i][j] = sparse.csr_matrix(np.identity(block_dims[dim_counter]))
                # B_temp[i][j] = id#'1'#sparse.csr_matrix(np.identity(block_dims[dim_counter]))
                # increment counter for next time, when the elif is satisfied
                dim_counter += 1
            elif i != 0 and j == 1:
                # M matrices
                B_temp[i - 1][j] = -1 * M_L_list[M_counter]
                B_temp[i][j] = -1 * M_R_list[M_counter]
                # B_temp[i-1][j] = id#'ML'#-1*M_L_list[M_counter]
                # B_temp[i][j] = id#'MR'#-1*M_R_list[M_counter]
                M_counter += 1
            # all other entries are 0 matrices
            # just need to get the dimensions right
            elif i <= 2 and j > 1:
                # zero matrices
                B_temp[i - 1][j] = np.zeros(
                    shape=(block_dims[int((i - 2) / 2)], block_dims[j])
                )
                B_temp[i][j] = np.zeros(
                    shape=(block_dims[int((i - 2) / 2)], block_dims[j])
                )
                # B_temp[i-1][j] = sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
                # B_temp[i][j] = sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
                # B_temp[i-1][j] = id#'0'#sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
                # B_temp[i][j] = id#'0'#sparse.csr_matrix(np.zeros(shape=(block_dims[int((i-2)/2)], block_dims[j])))
            else:

                # for i>=3, there will be a roll performed on the matrix, s. below.
                # Therfore the j index needs to be adjusted
                """
                print('i', i)
                print('j', j)
                print('counter ', block_dims_roll_counter)
                print('index ', int((i-2)/2))
                print('before roll ', block_dims_roll)
                """
                # only roll every second time
                # len(block_dims_roll) depends on N
                # the larger N, the more of j-indices must be considered
                if block_dims_roll_counter % int(len(block_dims_roll) - 2) == 0:
                    block_dims_roll = np.roll(block_dims_roll, -1)
                    # print('after roll ', block_dims_roll)

                # zero matrices
                B_temp[i - 1][j] = np.zeros(
                    shape=(block_dims[int((i - 2) / 2)], block_dims_roll[j])
                )
                B_temp[i][j] = np.zeros(
                    shape=(block_dims[int((i - 2) / 2)], block_dims_roll[j])
                )
                # print(B_temp[i-1][j].shape)
                # print(B_temp[i][j].shape)

                block_dims_roll_counter += 1

    # the entries of B_temp now must be sorted and put into a block matrix
    # e.g. for N=4 so far
    #        B_temp = ( T  0    0 )
    #                 ( 1  M_L  0 )
    #                 ( 1  M_R  0 )
    #                 ( 1  M_L  0 )
    #                 ( 1  M_R  0 )
    # as an array. But we want B as a matrix in the form
    #             B = ( T  0    0 )
    #                 ( 1  M_L  0 )
    #                 ( 1  M_R  0 )
    #                 ( 0  1  M_L )
    #                 ( 0  1  M_R )

    # therfore use np.roll() to bring stuff into right order
    roll_counter = 1
    for i in range(3, (2 * N - 3), 2):
        B_temp[i] = np.roll(B_temp[i], roll_counter)
        B_temp[i + 1] = np.roll(B_temp[i + 1], roll_counter)
        roll_counter += 1
    """
    print('[3][0]', B_temp[3][0].shape)
    print('[3][1]', B_temp[3][1].shape)
    print('[3][2]', B_temp[3][2].shape)
    print('[3][3]', B_temp[3][3].shape)
    print('[4][0]', B_temp[4][0].shape)
    print('[4][1]', B_temp[4][1].shape)
    print('[4][2]', B_temp[4][2].shape)
    print('[4][3]', B_temp[4][3].shape)
    print('[5][0]', B_temp[5][0].shape)
    print('[5][1]', B_temp[5][1].shape)
    print('[5][2]', B_temp[5][2].shape)
    print('[5][3]', B_temp[5][3].shape)
    print('[6][0]', B_temp[6][0].shape)
    print('[6][1]', B_temp[6][1].shape)
    print('[6][2]', B_temp[6][2].shape)
    print('[6][3]', B_temp[6][3].shape)
    """

    # finally define a block matrix using hstack and vstack
    s = B_temp.shape
    s0 = s[0]
    s1 = s[1]
    B_rows = []
    # b_test = np.block(B_temp[0])
    # print(b_test)
    for i in range(s0):
        B_rows.append(np.hstack(B_temp[i]).astype(np.float64))
    # print(B_rows)
    """
    print('len ', len(B_rows))
    print('B[0]', B_rows[0])
    print('B[0].shape', B_rows[0].shape)
    print('B[1].shape', B_rows[1].shape)
    print('B[2].shape', B_rows[2].shape)
    print('B[3].shape', B_rows[3].shape)
    print('B[4].shape', B_rows[4].shape)
    print('B[5].shape', B_rows[5].shape)
    print('B[6].shape', B_rows[6].shape)
    """
    # print(B_rows)
    B_matrix = np.vstack(B_rows)

    """
    TO DO: SPARSE MATRICES
    """

    return B_matrix, b_vector


"""
build c vector
needs as parameter the Hamiltonian + the x-vector and basis vector
"""


def Build_c_vector(ham, basis_vec, N):
    c_vec = []
    # total length of c vec
    total_length = sum([(2**i) * (2**i) for i in range(2, N + 1)])
    # print('x_vec ', x_vec)
    # print('len x_vec', len(x_vec_new))
    # for the c vector, we only need the first 16 components out of the
    # x-vector and basis vector, since
    # c^T * x = tr(rho^(2) * h)
    # depends on rho^(2), which only has 16 components
    ind = 16
    for i in range(0, ind):
        c_vec.append(np.trace((basis_vec[0][i].T) @ ham))

    for i in range(ind, total_length):
        c_vec.append(0)

    return c_vec

    #
