"""
This module contains functions (and classes) that help define the Hamiltonian of a simulated system
"""
import numpy as np
import scipy.sparse as sparse

# define spin operators and sparse them
def sx():
    sx = np.array([[0,1],[1,0]])/2
    #sxSp = sparse.csr_matrix(sx, shape=(2, 2))#, dtype=complex)
    #return sxSp
    return sx

sx=sx()
'''
def sy():
    sy = np.array([[0,-1j],[1j,0]])/2
    sySp = sparse.csr_matrix(sy, shape=(2, 2), dtype=complex)
    return sySp

sy=sy()
'''
def isy():
    isy = np.array([[0,1],[-1,0]])/2
    #isySp = sparse.csr_matrix(isy, shape=(2, 2))#, dtype=complex)
    #return isySp
    return isy
isy=isy()

def sz():
    sz = np.array([[1,0],[0,-1]])/2
    #szSp = sparse.csr_matrix(sz, shape=(2, 2))#, dtype=complex)
    #return szSp
    return sz

sz=sz()

def one():
    #oneSp = sparse.identity(2)
    oneSp = np.identity(2)
    return oneSp

Id=one()

'''
takes interaction term h and builds the full M-body Hamiltonian
'''
def Build_H_full(M, h, dim=2):
    H = 0
    k = M-1
    '''
    build the operator
    1 x 1 x ... x h x 1 x ... x 1
    where x denotes the kronecker product
    '''
    for i in range(0, k):
        H += np.kron(np.kron(np.identity(dim**i), h), np.identity(dim**(k-i-1)))

    return H


def Build_H_TFIM_Interaction_Term(h_z=1):
    # interaction term is
    # XX + h_z/2 (Z \otimes Id + Id \otimes Z)
    # extra 1/2 comes from symmertrizations
    interaction_term = -sparse.kron(sx, sx) - 1/2*(h_z/2)*(sparse.kron(sz, Id) + sparse.kron(Id, sz))
    return interaction_term

def Build_H_XXZ(Delta=1):
    # interaction term is
    #ham = sparse.kron(sx, sx) - sparse.kron(isy, isy) + Delta*sparse.kron(sz, sz)
    ham = np.kron(sx, sx) - np.kron(isy, isy) + Delta*np.kron(sz, sz)
    return ham

'''
M = N+1???
'''
def Build_H_XXZ_full(M, dim=2, Delta=1):
    H = 0
    ham = Build_H_XXZ(Delta=Delta)
    k = M-1
    '''
    build the operator
    1 x 1 x ... x h x 1 x ... x 1
    where x denotes the kronecker product
    '''
    op = []
    for i in range(0, k):
        operator = np.kron(np.kron(np.identity(dim**i), ham), np.identity(dim**(k-i-1)))
        op.append(operator)
    #for i in range(len(op)):
    #    print(op[i].shape)
    H = sum(op)
    return H

'''
get random normal distributed numbers put in a 4x4 matrix H
H' = H + H^T
is then a hermitian matrix
'''
def Build_random_H(seed):
    np.random.seed(seed)
    A = np.random.normal(size=(4, 4))
    H = 0.5*(A + A.T)
    return H


'''
define function that returns desired full Hamiltonian
'''
def get_Hamiltonian_full(model, N, *args):
    H = 0
    if model=='XXZ/':
        # expects Delta to be as optinal parameter
        for Delta in args:
            H = Build_H_XXZ_full(N-2, Delta=Delta)
    elif model=='random_Hamiltonian/':
        # expects seed to be an optinal parameter
        for seed in args:
            H = Build_random_H(seed)
            H = Build_H_full(N-2, H)
    else:
        print('Desired model not found.')
        print('Returning default Hamiltonian: XXZ for Delta = 1.')
        H = Build_H_XXZ_full(N-2, Delta=1)
    return H

'''
define function that returns desired Hamiltonian
'''
def get_Hamiltonian(model, *args):
    H = 0
    if model=='XXZ/':
        # expects Delta to be as optinal parameter
        for Delta in args:
            H = Build_H_XXZ(Delta=Delta)
    elif model=='random_Hamiltonian/':
        # expects seed to be an optinal parameter
        for seed in args:
            H = Build_random_H(seed)
    else:
        print('Desired model not found.')
        print('Returning default Hamiltonian: XXZ for Delta = 1.')
        H = Build_H_XXZ()
    return H


"""
We actually do not need the whole Hamiltonian, only the interaction term, that
stays the same due to LTI
"""
# function that builds all single spin operators
def Build_Single_Spin_Ops(N):
    sxi=[]
    syi=[]
    szi=[]

    #Id=one()
    # build the single spin operators
    for i in range(0, N):
        sxi.append(sparse.kron(sparse.kron(sparse.identity(2**(i)),sx), sparse.identity(2**(N-i-1))))
        syi.append(sparse.kron(sparse.kron(sparse.identity(2**(i)),sy), sparse.identity(2**(N-i-1))))
        szi.append(sparse.kron(sparse.kron(sparse.identity(2**(i)),sz), sparse.identity(2**(N-i-1))))

    return sxi, syi, szi

# Build the transverse field ising Hamiltonian with periodic boundaries
def Build_H_TFIM(N, h_z=1):
    dim = 2**N
    H = sparse.csr_matrix((dim,dim))
    sxis, syis, szis = Build_Single_Spin_Ops(N)
    # get the part of the Hamiltonians of neighbouring spins
    h_i = []
    #interaction_term = []
    for i in range(0, N):
        #interaction_term.append(-sxis[i] @ sxis[(i+1)%N])
        H = H - sxis[i] @ sxis[(i+1)%N] # interaction term
        H = H - h_z/2*szis[i] # field term
        h_i.append(H)
    return H, h_i #, interaction_term
