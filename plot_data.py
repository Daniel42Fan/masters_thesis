'''
this file contains functions to plot data
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pickle as pickle

import define_system as ds
import calc_energies as ce
import analyze_data as ad
import solve_SDP as ssdp


plt.rcParams.update({
"text.usetex": True,
#"font.family": "Helvetica"
})
'''
create a template for plots
'''
def create_plot_template():
    fig, ax = plt.subplots(figsize=((10,6)))
    #fig, ax = plt.subplots()
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    #plt.rcParams['text.usetex'] = True
    return fig, ax

'''
creates heat maps
'''
def heat_maps(data):

    return plt

#e_before_gd, e_after_gd = get_energies(0, 5, 2)
'''
def get_data_w_settings():
    return e_before_gd, e_after_gd
'''
def compare_energies_grad_descent(xdata, ydata):
    fig, ax = plt.subplots(figsize=((16,9)))
    #ax.plot(xdata, ydata, marker='x', linestyle='None')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    plt.show()
    return fig, ax

'''
Plot energies
Decide, whether to use normalized energies or not
'''
def plot_energies(model, map, *args, normalize=True, xscale='log', yscale='log'):
    markers = ['.', '1', 'v', 'p', '*', 'x', '+', 's']
    model = model
    map = map
    e_before_gd_list = []
    e_after_gd_list = []
    chi_list = []
    figs_list = []
    axes_list = []
    for Delta in range(1, 2):
        for N in range(5, 7, 2):
            # calc the energies for no cg
            '''
            TO DO: need a better way to extract Hamiltonian!!!!!
            '''
            #prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(N-1, N, args[1], dim=2, output=False)
            prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(N-1, N, args[0], dim=2, output=True)
            e_N_minus_1 = prob_N_minus_1.value
            e_N = prob_N.value
            de = e_N-e_N_minus_1
            #try:
            for chi in range(2, 8):
                # get data
                e_before_gd, e_after_gd = ad.get_energies(Delta, N, chi, model=model, map=map, normalize=normalize)
                if normalize==True:
                    e_after_gd = 1-np.asarray(e_after_gd)
                    e_before_gd = 1-np.asarray(e_before_gd)
                else:
                    e_after_gd = e_N-np.asarray(e_after_gd)
                    e_before_gd = e_N-np.asarray(e_before_gd)
                    #e_after_gd = (e_N+1e-06) - np.asarray(e_after_gd)
                    #e_before_gd = (e_N+1e-06) - np.asarray(e_before_gd)
                e_before_gd_list.append(e_before_gd)
                e_after_gd_list.append(e_after_gd)
                chi_list.append(chi)
            #print('Plotting energies for Delta = {}, N = {}, chi = {}'.format(Delta, N, chi))
            print('Plotting energies for Delta = {}, N = {}'.format(Delta, N))
            fig, ax = create_plot_template()
            for i in range(len(e_after_gd_list)):
                #print('before gd\n', e_before_gd_list[i])
                #print('after gd\n', e_after_gd_list[i])
                ax.plot(e_before_gd_list[i], e_after_gd_list[i], marker=markers[i],
                        linestyle='None', label=r'$\chi = {}$'.format(chi_list[i]))
            plt.legend(loc='upper left')
            '''
            TEST THE SIGNIFICANT NUMBERS AND FORMAT
            '''
            plt.title(r'$\Delta E = {0:.2e}$'.format(de), loc='left')
            # extract seed out of args
            # if random Hamiltonian: get seed name
            #plt.title('seed = {}'.format(args[0]), loc='right')
            path = ce.create_path('plots/energies/'+model+map)
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)
            #plt.xscale(xscale)
            #plt.yscale(yscale)
            '''
            x_left, x_right = plt.xlim()
            y_lower, y_upper = plt.ylim()
            # distance between left and right corners
            x_dis = x_right - x_left
            y_dis = y_upper - y_lower
            # add vertical and horizontal lines with energies E_4 and E_5
            ax.hlines(y=e_N, xmin=-1, xmax=0, colors='k',
                        linestyle='--', linewidth=1)#, label='E_{}'.format(N))
            ax.text(x_left+0.015*(x_dis), e_N+0.01*(y_dis), r'$E_{}$'.format(N),
                    fontsize='x-large', color='k')
            ax.vlines(e_N_minus_1, ymin=-1, ymax=1 , colors='olive',
                        linestyles='--', linewidth=1)#, label='E_{}'.format(N-1))
            ax.text(e_N_minus_1+0.01*(x_dis), y_lower+0.02*(y_dis), r'$E_{}$'.format(N-1),
                    fontsize='x-large', color='olive')
            ax.vlines(e_N, ymin=-1, ymax=1 , colors='gray',
                        linestyles='--', linewidth=1)#, label='E_{}'.format(N))
            ax.text(e_N+0.01*(x_dis), y_lower+0.02*(y_dis), r'$E_{}$'.format(N),
                    fontsize='x-large', color='gray')
            plt.xlim(x_left, x_right)
            plt.ylim(y_lower, y_upper)
            '''

            '''
            CHANGE TITLE NAME ACCORDINGLY!!!
            '''
            if normalize==True:
                ax.set_title('Normalized Energies for a Random Hamiltonian\nN = {}, Delta = {}'.format(N, Delta))
                ax.set_xlabel('Accuracy Before Gradient Descent')
                ax.set_ylabel('Accuracy After Gradient Descent')
                plt.savefig(path+'normalized_energies_Delta{}_N{}.pdf'.format(Delta, N))
            else:
                #ax.set_title('Energies for the XXZ Hamiltonian $\Delta = {}$\nN = {}'.format(Delta, N))
                ax.set_title('Energies for a Random Hamiltonian\nN = {}'.format(N))
                #ax.set_xlabel('Energy Before Gradient Descent')
                #ax.set_ylabel('Energy After Gradient Descent')
                ax.set_xlabel('Accuracy Before Gradient Descent')
                ax.set_ylabel('Accuracy After Gradient Descent')
                #plt.savefig(path+'energies_Delta{}_N{}.pdf'.format(Delta, N))
                plt.savefig(path+'energies_N{}.pdf'.format(N))
            print('Plot saved successfully')
            figs_list.append(fig)
            axes_list.append(ax)
            plt.close("all")
            #plt.show()
            #except:
            #    print('No data found')
    #return 0
    return figs_list, axes_list
'''
seeds = [1,2,4,5,7,8]
for s in seeds:
#for s in range(1,11):
    #plot_energies('random_Hamiltonian/seed{}/'.format(s), 'random_maps/')
    ham = ds.Build_random_H(s)
    scale = 'log'
    figs, axes = plot_energies('random_Hamiltonian/seed{}/'.format(s), 'random_maps/', s, ham,
                                normalize=False, xscale=scale, yscale=scale)
'''
'''
ham = ds.Build_H_XXZ(Delta=1)
figs, axes = plot_energies('XXZ/', 'random_maps/', ham, normalize=False)
'''

def plot_heat_maps():
    model = 'XXZ/'
    '''
    Delta = 1
    N = 5
    chi = 3
    '''
    #fig, ax = plt.subplots(figsize=((16,9)))
    for Delta in range(0, 3):
        for N in range(7, 9, 2):
            for chi in range(1, N):
                try:
                    W = ad.get_optimized_cg_map(Delta, N, chi)
                    a = ad.get_subspace_angles(W)
                    print('Plotting heatmaps for Delta = {}, N = {}, chi = {}'.format(Delta, N, chi))
                    plt.imshow(a, cmap='inferno', interpolation='nearest')
                    plt.colorbar()
                    plt.title('Subspace angles between different coarse graining maps for\n\Delta = {}, N = {}, \chi = {}'.format(Delta, N, chi))
                    #plt.xlabel('')
                    #plt.ylabel('')
                    plt.savefig('plots/subspace_angles/'+model+'subspace_angles_Delta{}_N{}_chi{}.pdf'.format(Delta, N, chi))
                    plt.close("all")
                except:
                    print('Map not found')


    return 0

#plot_heat_maps()

def plot_subspace_energies(model, map, *args):
    model = model
    map = map
    for Delta in range(0, 1):
        for N in range(5, 7, 2):
            ham = ds.get_Hamiltonian_full(model, N, args)
            fig, ax = create_plot_template()
            labels = ['label 1', 'label 2']
            labels0 = ['test 1', 'test 2']
            for chi in range(2, 3):
                #try:
                # get data
                eigvals, eigvals_H, spectrum, eigval0, spectrum0 = ad.calc_q_dagger_H_q(Delta, N, chi, ham, model, map, *args)
                # Extract first and second entries separately
                entries = []
                entries0 = []
                print('len(spectrum[0]) = ', len(spectrum[0]))
                for i in range(len(spectrum[0])):
                    e = [entry[i] for entry in spectrum]
                    entries.append(e)
                for i in range(len(spectrum0[0])):
                    e = [entry[i] for entry in spectrum0]
                    entries0.append(e)
                '''
                first_entries = [entry[0] for entry in spectrum]
                second_entries = [entry[1] for entry in spectrum]
                first_entries0 = [entry[0] for entry in spectrum0]
                second_entries0 = [entry[1] for entry in spectrum0]
                '''
                print('Plotting energies for Delta = {}, N = {}, chi = {}'.format(Delta, N, chi))
                ax.set_title('$N = {}, \Delta = {}, \chi = {}$'.format(N-2, Delta, chi))
                labels = ['test1', 'test2', 'test3']
                labels0 = ['test01', 'test02', 'test03']
                for i in range(len(spectrum[0])):
                    ax.plot(entries[i], marker='x', linestyle='None', label=labels[i])
                    ax.plot(entries0[i], marker='*', linestyle='None', label=labels0[i])
                '''
                ax.plot(first_entries, marker='x', linestyle='None', label='test 1')
                ax.plot(second_entries, marker='x', linestyle='None', label='test 2')
                ax.plot(first_entries0, marker='*', linestyle='None', label='test 3')
                ax.plot(second_entries0, marker='*', linestyle='None', label='test 4')
                '''
                ax.hlines(y=eigvals_H, xmin=0, xmax=len(eigvals), colors='k', linestyle='--', label='spectrum of H')
            ax.set_xlabel('Random Coarse-Grained Maps $W$')
            ax.set_ylabel('Energies')
            plt.legend()
            #plt.legend([a], ['min eigval','max eigval'])
            #path = ce.create_path('plots/subspace_energies/'+model+map)
            #plt.savefig(path+'energies_Delta{}_N{}.pdf'.format(Delta, N))
            #print('saved')
            #plt.close("all")
            plt.show()
                    #except:
                    #    print('No data found')
    return 0

'''
seeds  = [3,6,9]
ham = ds.Build_random_H(3)
ham_full = ds.Build_H_full(N-2, ham)
'''

model = 'XXZ/'
plot_subspace_energies(model, 'random_maps/', 0)

#model = 'random_Hamiltonian/'
#plot_subspace_energies(model, 'random_maps/', 3)

'''
plot the intermediate results from the gradient descent minimization
'''
def plot_grad_descent_intermediate_results():
    model = 'XXZ/'
    map = 'random_maps/'
    markers = ['o', 's', '^', 'D', 'v', 'p', '>', '<', '*', 'H']
    for Delta in range(0, 1):
        ham = ds.Build_H_XXZ(Delta=Delta)
        for N in range(5, 7, 2):
            for chi in range(2, 3):
                #try:
                prob_N_minus_1, prob_N = ssdp.compare_solutions_no_cg(N-1, N, ham, dim=2, output=True)
                e_N_minus_1 = prob_N_minus_1.value
                e_N = prob_N.value
                # get result
                history = ad.get_history(Delta, N, chi, model=model, map=map, normalize=False)
                print('Plotting intermediate energy values of gradient descent for Delta = {}, N = {}, chi = {}'.format(Delta, N, chi))
                fig, ax = create_plot_template()
                ax.set_title('Intermediate Energy Values of the Gradient Search for the XXZ Model\nN = {}, $\Delta = {}$, $\chi = {}$'.format(N, Delta, chi))
                # plot first 10 maps
                for i in range(10):
                    #ax.plot(-1*np.asarray(history[i]), marker='x', linestyle='None')#, label='energies')
                    print('markers[{}]'.format(i), markers[i])
                    ax.plot(-1*np.asarray(history[i]), marker=markers[i], label='Map {}'.format(i+1))
                # only integer ticks on x-axis
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                x_left, x_right = plt.xlim()
                y_lower, y_upper = plt.ylim()
                # distance between left and right corners
                x_dis = x_right - x_left
                y_dis = y_upper - y_lower
                ax.hlines(y=e_N, xmin=-1, xmax=0, colors='k',
                            linestyle='--', linewidth=1)#, label='E_{}'.format(N))
                ax.text(x_left+0.015*(x_dis), e_N+0.01*(y_dis), r'$E_{}$'.format(N),
                        fontsize='x-large', color='k')
                ax.hlines(y=e_N_minus_1, xmin=-1, xmax=0, colors='m',
                            linestyle='--', linewidth=1)#, label='E_{}'.format(N))
                ax.text(x_left+0.015*(x_dis), e_N_minus_1+0.01*(y_dis), r'$E_{}$'.format(N-1),
                        fontsize='x-large', color='m')
                ax.set_xlabel('Step')
                ax.set_ylabel('Energies')
                plt.xlim(-1, 42)
                plt.legend(loc=4) # lower right position
                #path = ce.create_path('plots/history/'+model+map)
                #plt.savefig(path+'history_Delta{}_N{}_chi{}.pdf'.format(Delta, N, chi))
                #print('plot saved successfully')
                #plt.close("all")
                plt.show()
                #except:
                #    print('No data found')
    return 0

#plot_grad_descent_intermediate_results()

def plot_compare_g_tol(model, map):
    model= model
    map = map
    #g_tol = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    g_tol = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]#, 1e-7, 1e-8]
    e_before_gd_list = []
    e_after_gd_list = []
    for Delta in range(1, 2):
        for N in range(7, 9, 2):
            for chi in range(1, N):
                try:
                    for g in g_tol:
                        g = 'g_tol'+str(g)+'/'
                        e_before_gd, e_after_gd = ad.get_energies(Delta, N, chi, g, model=model, map=map)
                        e_after_gd = 1-np.asarray(e_after_gd)
                        e_before_gd = 1-np.asarray(e_before_gd)
                        e_before_gd_list.append(e_before_gd)
                        e_after_gd_list.append(e_after_gd)
                    #print('e_before_gd_list =\n', e_before_gd_list)
                    #print('e_after_gd_list =\n', e_after_gd_list)
                    fig, ax = create_plot_template()
                    for i in range(len(e_before_gd_list)):
                        ax.plot(e_before_gd_list[i], e_after_gd_list[i], marker='x', linestyle='None', label='g_tol = {}'.format(g_tol[i]))
                    ax.set_title('Comparing gtol parameter for\nN = {}, Delta = {}, chi = {}'.format(N, Delta, chi))
                    ax.set_xlabel('1- Normalized Energy Before Gradient Descent')
                    ax.set_ylabel('1- Normalized Energy After Gradient Descent')
                    plt.xscale('log')
                    plt.yscale('log')
                    plt.legend()
                    path = ce.create_path('plots/energies_compare_gtol/'+model)
                    #path = ce.create_path('plots/energies/'+model+'pertubation/')
                    plt.savefig(path+'energies_Delta{}_N{}_chi{}.pdf'.format(Delta, N, chi))
                    print('Plot saved successfully')
                    plt.close("all")
                except:
                    print('No data found')
    return 0

#plot_compare_g_tol('XXZ/', 'random_maps/')
#for s in range(1,11):
#    plot_compare_g_tol('random_Hamiltonian/seed{}/'.format(s), 'random_maps/')


#h = ad.get_res_history()
#print(h)


def test_boxplots():
    standard_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # Generate some random data
    data1 = np.random.normal(0, 1, 100)
    data2 = np.random.normal(1, 1, 100)
    data3 = np.random.normal(2, 1, 100)
    data4 = np.random.normal(2, 1, 100)

    # Create subplots
    fig, ax = plt.subplots(figsize=(10, 6))

    # Combine the data sets
    data = [[data1,data1,data1], [data2,data2,data2], [data3,data3,data3], [data4,data4,data4]]
    # Define the positions for the boxplots
    '''
    positions1 = [1, 4, 7]
    positions2 = [2, 5, 8]
    positions3 = [3, 6, 9]
    positions4 = [10, 11, 12]
    positions = [positions1, positions2, positions3, positions4]
    '''
    positions = [[1,2,3],[1,2,3],[1,2,3],[1,2,3]]
    '''
    # Create a boxplot on the Axes with custom labels
    chi_label = ['$\chi = {}$'.format(i) for i in range(2, 8)]
    ax.boxplot(data, labels=chi_label)
    '''

    # Set colors for the boxplots
    #colors = ['red', 'green', 'blue']
    #for patch, color in zip(bp['boxes'], colors):
    #    patch.set_facecolor(color)
    # Iterate over the data sets and their positions
    for i, (data_set, pos) in enumerate(zip(data, positions)):
        # Create boxplot for each data set with specified position and color
        bp = ax.boxplot(data_set, positions=pos, patch_artist=True)

        # Set color for the box, whiskers, maximum, and minimum
        for key in bp.keys():
            if isinstance(bp[key], list):
                for component in bp[key]:
                    #if key in ['boxes', 'whiskers', 'caps', 'medians']:
                    if key in ['boxes', 'whiskers', 'caps']:
                        component.set(color=standard_colors[i], linewidth=1)



    #ax.set_yscale('log')

    # Set labels on the Axes
    ax.set_xlabel('Groups')
    ax.set_ylabel('Values')

    # Show the plot
    plt.show()

    return 0

#test_boxplots()



#
