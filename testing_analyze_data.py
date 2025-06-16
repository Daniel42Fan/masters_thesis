"""
testing the functions from analyze_data.py
"""

import analyze_data as ad
import define_system as ds


def testing_normalized_energies_to_absolute_values(
    Delta, N, chi, ham, *args, dim=2, model="XXZ/", map="", normalize=True
):
    e_before_gd, e_after_gd = ad.normalized_energies_to_absolute_values(
        Delta, N, chi, ham, *args, dim=dim, model=model, map=map, normalize=normalize
    )
    return e_before_gd, e_after_gd


"""
ham_seeds = [3,6,9]
#ham = ds.Build_random_H(seed_ham)
ham = ds.Build_random_H(3)
seed = 1
e_before_gd, e_after_gd = testing_normalized_energies_to_absolute_values(1, 5, 2, ham,
                                dim=2, model='random_Hamiltonian/seed{}/'.format(seed), map='random_maps/', normalize=True)

print('e_before_gd =\n', e_before_gd)
print('e_after_gd =\n', e_after_gd)
"""


def testing_save_converted_energies(
    Delta, N, chi, ham, *args, dim=2, model="XXZ/", map="", normalize=True
):
    ad.save_converted_energies(
        Delta, N, chi, ham, *args, dim=dim, model=model, map=map, normalize=normalize
    )
    return 0


"""
N=5
Delta=1
#ham_seeds = [3,6,9]

for seed_ham in range(1,11):
    ham = ds.Build_random_H(seed_ham)
    for chi in range(5,8):
            testing_save_converted_energies(Delta, N, chi, ham,
                    dim=2, model='random_Hamiltonian/seed{}/'.format(seed_ham), map='random_maps/', normalize=True)
"""
"""
ham = ds.Build_H_XXZ(Delta=Delta)
for chi in range(1,5):
        testing_save_converted_energies(Delta, N, chi, ham,
                dim=2, model='XXZ/', map='', normalize=True)
"""


def testing_get_optimized_cg_map(
    Delta, N, chi, dim=2, *args, model="XXZ/", map="", normalize=False
):
    W = ad.get_optimized_cg_map(
        Delta, N, chi, dim=dim, *args, model=model, map=map, normalize=normalize
    )
    return W


"""
N = 7
Delta = 0
chi = 3
W = testing_get_optimized_cg_map(Delta, N, chi, map='random_maps/')
print('Optimized maps =\n', W)
"""


def testing_investigate_optimized_maps(
    Delta, N, chi, dim=2, *args, model="XXZ/", map="", normalize=False
):
    x = ad.investigate_optimized_maps(
        Delta, N, chi, dim=dim, *args, model=model, map=map, normalize=normalize
    )
    return 0


"""
testing_investigate_optimized_maps(Delta, N, chi, dim=2, model='XXZ/',
                                    map='random_maps/', normalize=False)
"""


def testing_get_error_text(Delta, N, chi, *args, model="XXZ/", map="", normalize=False):
    errors = ad.get_error_txt(
        Delta, N, chi, *args, model=model, map=map, normalize=normalize
    )
    numbers = ad.extract_error_seeds(errors)
    return numbers


N = 5
Delta = 0
chi = 2
map = "random_maps/"

n = testing_get_error_text(Delta, N, chi, model="XXZ/", map=map)
print(n)


#
