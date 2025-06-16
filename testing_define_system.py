'''
testing the functions from define_system.py
'''

import define_system as ds

def testing_Build_random_H(seed):
    H = ds.Build_random_H(seed)
    print('Random Hamiltonian =\n', H)
    print('Test for hermitian:\n', H == H.conj().T)
    return 0

'''
testing_Build_random_H(0)
testing_Build_random_H(0)
testing_Build_random_H(0)
testing_Build_random_H(2)
testing_Build_random_H(3)
'''









































#
