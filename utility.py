"""
This module contains utility functions
"""
import numpy as np
import cvxpy as cp
from numpy import linalg as nLA
import scipy.sparse as sparse
import define_SDP_cg as dsdp_cg

'''
sorts eigenvalues (+its corresponding eigenvectors) from small to large
returns n eigenvalues, eigenvectors
'''
def eigen(A, n):
    eigenValues, eigenVectors = nLA.eig(A)
    idx = np.argsort(eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues[:n], eigenVectors[:,:n])


'''
defines function for partial trace
'''
def ptrace(matrix, dims: tuple[int], system):
    # parameter dims:
    # for a tensor space of dimesnions n \otimes m
    # we need as input dims = (n,m),
    # where n is the dim of the first subsystem,
    # m is dimension of the second subsystem

    # default axis
    axis1 = 0
    axis2 = 2
    # reshape
    n = dims[0]
    m = dims[1]
    if system == 1:
        axis1=0
        axis2=2
    elif system == 2:
        axis1=1
        axis2=3
    else:
        print("wrong system")
        #return 0

    A = matrix.reshape((n,m,n,m))
    ptr = np.trace(A, axis1=axis1, axis2=axis2)
    #ptr = ptr.H
    return ptr

'''
adapted from the qutipy package, so it can handle cp variables without
converting it into np
UNFINISHED!!!
'''
def partial_trace(X, sys, dim):
    """
    sys is a list of systems over which to take the partial trace (i.e., the
    systems to discard).

    Example: If rho_AB is a bipartite state with dimA the dimension of system A
    and dimB the dimension of system B, then

    partial_trace(rho_AB,[2],[dimA,dimB]) gives the density matrix on

    system A, i.e., rho_A:=partial_trace[rho_AB].

    Similarly, partial_trace(rho_AB,[1],[dimA,dimB]) discards the first subsystem,
    returning the density matrix of system B.

    If rho_ABC is a tripartite state, then, e.g.,

    partial_trace(rho_ABC,[1,3],[dimA,dimB,dimC])

    discards the first and third subsystems, so that we obtain the density
    matrix for system B.

    """

    if isinstance(X, cvxpy.Variable):
        X = cvxpy_to_numpy(X)
        X_out = partial_trace(X, sys, dim)
        return numpy_to_cvxpy(X_out)

    if not sys:  # If sys is empty, just return the original operator
        return X
    elif len(sys) == len(dim):  # If tracing over all systems
        return Tr(X)
    else:
        if X.shape[1] == 1:
            X = X @ dag(X)

        num_sys = len(dim)
        total_sys = range(1, num_sys + 1)

        dims_sys = [
            dim[s - 1] for s in sys
        ]  # Dimensions of the system to be traced over
        dims_keep = [dim[s - 1] for s in list(set(total_sys) - set(sys))]
        dim_sys = np.product(dims_sys)
        dim_keep = np.product(dims_keep)

        perm = sys + list(set(total_sys) - set(sys))
        X = syspermute(X, perm, dim)

        X = np.array(X)
        dim = [dim_sys] + dims_keep
        X_reshape = np.reshape(X, dim + dim)
        X_reshape = np.sum(np.diagonal(X_reshape, axis1=0, axis2=len(dim)), axis=-1)
        X = np.reshape(X_reshape, (dim_keep, dim_keep))

        return X

'''
adapted from the qutipy package, so it can handle cp variables without
converting it into np
UNFINISHED!!!
'''
def syspermute(X, perm, dim):
    """
    Permutes order of subsystems in the multipartite operator X.

    perm is a list
    containing the desired order, and dim is a list of the dimensions of all
    subsystems.
    """

    # If p is defined using np.array(), then it must first be converted
    # to a numpy array, or else the reshaping below won't work.
    #X = np.array(X)

    n = len(dim)
    d = X.shape

    perm = np.array(perm)
    dim = np.array(dim)

    if d[0] == 1 or d[1] == 1:
        # For a pure state
        perm = perm - 1
        tmp = cp.reshape(X, dim)
        q = cp.reshape(np.transpose(tmp, perm), d)

        return q
    elif d[0] == d[1]:
        # For a mixed state (density matrix)
        perm = perm - 1
        perm = np.append(perm, n + perm)
        dim = np.append(dim, dim)
        # cast into a tuple
        dim = tuple(dim)
        print('dim = ', dim)
        print('dim type = ', type(dim))
        tmp = cp.reshape(X, dim)
        Y = cp.reshape(np.transpose(tmp, perm), d)

        return Y

'''
from the qutipy package
'''
def cvxpy_to_numpy(cvx_obj):
    """
    Converts a cvxpy variable into a numpy array.
    """

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


# depending on a matrix, get all possible combination of tuples of indices
# e.g. (0,0), (0,1), (1,0), (1,1)
def get_index_tuple(mat):
    # stuff
    return 0

'''
needed for builiding matrix reps
e.g. linear_op(basis) = V * basis * V_dagger
*args checks for V_L or V_R
expects a string 'L' or 'R'
'''
def linear_op_cg_maps(basis, *args):#, grad_descent=False, index_tuple=(0,0), roll=0):
    '''
    this can be done better with keyword arguments
    '''
    arg = args[0]
    # the kwargs are stored in a dictionary.
    # it is empty, if we do not consider gradient descent
    kwargs = args[1]
    # get the parameters from get_V_matrices function, a little bit hard coded
    dim = arg[1]
    chi = arg[2]
    # need to put W as an argument
    W = arg[3]

    # grad_descent, index_tuple, roll are all stored in kwargs,
    # IF we consider grad descent
    # initilaize these parameters regardless
    grad_descent=False
    index_tuple=(0,0)
    roll=0
    # check if dictionary is empty
    if not kwargs:
        #do nothing
        pass
    else:
        grad_descent = kwargs['grad_descent']
        index_tuple = kwargs['index_tuple']
        roll = kwargs['roll']

    # initialize
    mat = 0
    # for product rule in the derivative there are two parts of the sum
    mat_1 = 0
    mat_2 = 0

    V_L = 0
    V_L_dagger = 0
    V_R = 0
    V_R_dagger = 0
    if grad_descent==False:
        #print('grad_descent false')
        V_L, V_L_dagger, V_R, V_R_dagger = dsdp_cg.cg_maps(W, dim=dim, chi=chi)

        # since we got a tuple as an optional argument,
        # this tuple is stored insinde another tuple
        # So we have to get the first element out of it
        for a in arg[0]:
            if (a == 'L'):
                mat = V_L @ basis @ V_L_dagger
                return mat
            if (a == 'R'):
                mat = V_R @ basis @ V_R_dagger
                return mat

    elif grad_descent==True:
        #print('arg[0] = ', arg[0])
        #print('index_tuple', index_tuple)
        #print('roll = ', roll)
        '''
        #W = dsdp_cg.generate_linear_map(dim=dim, chi=chi)
        if (arg[0] == 'L'):
            V_L, V_L_dagger = dsdp_cg.cg_maps_grad_descent(W, index_tuple, dim=dim, chi=chi,
                                roll=roll, L=True, grad_descent=grad_descent)
        if (arg[0] == 'R'):
            V_R, V_R_dagger = dsdp_cg.cg_maps_grad_descent(W, index_tuple, dim=dim, chi=chi,
                                roll=roll, R=True, grad_descent=grad_descent)
        '''
        '''
        E_L/E_R stuff hinzufuegen
        '''
        V_L, V_L_dagger, V_R, V_R_dagger = dsdp_cg.cg_maps(W, dim=dim, chi=chi)
        #print('V_L', V_L)
        #print('V_L_dagger', V_L_dagger)
        E_L, E_L_dagger, E_R, E_R_dagger = dsdp_cg.get_E_matrices(W, index_tuple, dim=dim)
        #print('E_L = ', E_L)
        #print('E_L_dagger = ', E_L_dagger)

        for a in arg[0]:
            if (a == 'L'):
                #print('roll = ', roll)
                if (roll==0):
                    mat = E_L @ basis @ V_L_dagger
                else:
                    mat = V_L @ basis @ E_L_dagger
                return mat
            if (a == 'R'):
                #print('roll = ', roll)
                if (roll==0):
                    mat = E_R @ basis @ V_R_dagger
                else:
                    mat = V_R @ basis @ E_R_dagger
                return mat



    # reshape
    # l = l.transpose(2,0,1).reshape(-1,l.shape[1])
    print('No optional arguments given')
    return mat

def linear_op_ptr(basis, *args):
    # extract dims and system out of the args for the ptr function
    arg = args[0]
    dims = arg[0]
    system = arg[1]
    # calc the ptr
    res = ptrace(basis, dims, system)
    return res


'''
builds the matrices for the constraints like
V rho V_dagger = ptr(omega)

Tests, if everything works is in the testing_define_SDP_cg.py file
'''
def Build_matrix_rep(lin_op, basis_in, basis_out, *args, **kwargs):
    '''
    this can be done better with additional keyword arguments
    '''
    # maybe the names of rows and columns are actually mixed up, but the code
    # works for now as it is
    # get the dimensions from the shape of the basis matrices
    #print('basis_in', basis_in)
    print('basis_in.shape', basis_in[0].shape)
    #print('basis_out', basis_out)
    print('basis_out.shape', basis_out[0].shape)
    n_rows = np.prod(basis_in[0].shape)
    n_cols = np.prod(basis_out[0].shape)
    #print('n_rows', n_rows)
    # initialize matrix. Shape is like the basis out
    mat = np.zeros((n_cols, n_rows))
    #print('n_cols', n_cols)
    #print('n_rows', n_rows)

    for i in range(0, n_cols):
        for j in range(0, n_rows):
            #print('({}, {})'.format(i,j))
            # we pass a tuple of optional arguments into lin_op.
            # So within lin_op, we have this tuple inside another tuple
            # Take the first element of the new tuple in the lin_op function
            mat[i][j] = np.trace(lin_op(basis_in[j], args, kwargs).conj().T @ basis_out[i])
    return mat

'''
define function that creates a single basis matrix in a certain shape
in the standard basis with only 0 and a single 1 at a given index position
'''
def create_basis_matrix(shape_, index_tuple):
    j, i = index_tuple
    E = np.zeros(shape_)
    #print('shape_ = ', shape_)
    E[j][i] = 1
    E = sparse.csr_matrix(E, shape=shape_)
    #print('E.shape = ', E.shape)
    return E


'''
defines a test function, to calculate its derivative
the function looks like f(X) = tr(X*M)
'''
def func_test(*args):
    # we expect here in the test function the first argument to be X
    X = 0
    if type(args[0]) is tuple:
        #print('tuple')
        X = args[0][0]
    else:
        #print(' no tuple')
        X = args[0]
    print('X', type(X))
    # get the mxn shape of X
    m = X.shape[0]
    n = X.shape[1]
    s = (m,n)
    # for the new M matrix, it must be a nxm matrix.
    # get a fixed random seed
    np.random.seed(42)
    M = np.random.randint(10, size=(n*m))
    # reshape
    M = M.reshape((n, m))
    print('X =\n', X)
    print('M =\n', M)
    print('calculate the trace of X@M \n', X@M)
    # finally, define the function
    func = np.trace(X@M)

    # return also the shape of input matrix
    return func, s


'''
Define a function that calculates the derivative of a function in the form
f(X) = tr(X*M)
This is used to understand, how to calculate more complicated derivatives for
our problem

func is a function, index_tuple (i,j) is needed to calculate the derivative of
df/d(x_ij)
*args has all arguments for a specific function
'''
def calc_derivative_test(func, index_tuple, *args):
    f, shape = func(args)
    # get a matrix basis, whcih is neede for the derivative
    basis = create_basis_matrix(shape, index_tuple)
    print('basis', basis)
    print('f', f)
    print('func', func)
    # calc the derivative
    deriv, _ = func(basis)
    return deriv

'''
for each index_tuple the result is a single number of an entry of a matrix.
Therfore we need to loop through every possible index to build the whole matrix.

HAVENT CHECKED IF SCALABLE
'''
def get_derivative_V(N, W, dim=2, chi=3):
    # initialize default parameters
    grad_descent=True
    index_tuple=(0,0)
    # get cg map
    #print('used W map in get derivative V\n', W)
    W = W
    k = W.shape[0]
    l = W.shape[1]
    #print('k = ', k)
    #print('l = ', l)

    indices = []
    # create all possible indices of the W matrix
    for i in range(0, k):
        for j in range(0, l):
            indices.append((i,j))

    V_L_derivative = []
    V_R_derivative = []
    V_L_list = []
    V_R_list = []

    # ini array to store individual matrix for the derivative at (i,j)
    dV_L_mat = np.empty(shape=(k, l), dtype='object')
    dV_R_mat = np.empty(shape=(k, l), dtype='object')

    # the product rule will be considered, when using the roll parameter,
    # to get the second term
    '''
    NAME OF THE roll PARAMETER OUTDATED, BUT STILL DOES ITS JOB
    '''
    for ind in indices:
        #print('indices' ,ind)
        #roll = 0
        V_L, V_R = dsdp_cg.get_V_matrices(N, W, dim=dim, chi=chi, grad_descent=grad_descent,
                                            index_tuple=ind, roll=0)
        # V_L is already a list, so extract the array out of it
        #V_L_list.append(V_L[0])
        #V_R_list.append(V_R[0])
        # now for the product rule, set the roll parameter to 1
        #roll = 1
        V_L2, V_R2 = dsdp_cg.get_V_matrices(N, W, dim=dim, chi=chi, grad_descent=grad_descent,
                                            index_tuple=ind, roll=1)

        V_L_tot = (V_L[0] + V_L2[0])
        V_R_tot = (V_R[0] + V_R2[0])
        # apply product rule here
        V_L_list.append(V_L_tot)
        V_R_list.append(V_R_tot)

        #print(f'dV matrix at index {ind}\n', V_L[0] + V_L2[0])
        #print('ind = ', ind)
        i = ind[0]
        j = ind[1]
        dV_L_mat[i][j] = V_L_tot
        dV_R_mat[i][j] = V_R_tot

    # finally the derivative is the sum of all matrices in the list, bc we have
    # to sum over all possible indices
    V_L_derivative.append(sum(V_L_list))
    V_R_derivative.append(sum(V_R_list))

    # get derivative at point (i, j)
    #dV_L_mat = np.reshape(V_L_list, (k,l))
    #dV_R_mat = np.reshape(V_R_list, (k,l))

    return V_L_derivative, V_R_derivative, dV_L_mat, dV_R_mat
    '''
    print('PRINTING V_L_list')
    print('len V_L_list', len(V_L_list))
    for i in range(0, len(V_L_list)):
        print(V_L_list[i])
        print('V_L_list.shape', V_L_list[i].shape)
    print('END PRINTING V_L_list')

    return V_L_list, V_R_list
    '''

















#
