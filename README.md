# Code of my Master's Thesis
[Link](https://utheses.univie.ac.at/detail/70909/) to my written thesis.
## Thesis title: "Optimizing constraints for ground state optimization"
The aim of my thesis was to use tools from mathematical optimization to numerically compute a quantum mechanical quantity that could not be solved analytically.

In more technical terms: lower bounds on the ground state energy of the many-body ground state problem were optimized for local Hamiltonians on an 1D spin chain with nearest-neighbor interaction. 
Using a suggested [relaxation method](https://arxiv.org/abs/2212.03014), the constraints of the ground state problem were compressed by using coarse-graining maps to reduce the dimension of the problem.
With a gradient-based method, these lower bounds on the ground state energy were optimized starting from randomly chosen coarse-graining maps.

## Structure of my code
* [utility.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/utility.py): Contains utility functions.
* [define_system.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/define_system.py): Defines and builds physical systems where optimizations of the lower bounds on the ground state energy are carried out.
* [define_SDP.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/define_SDP.py)/[define_SDP_cg.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/define_SDP_cg.py): The problem of my thesis can be formulated as a [semidefinite program](https://en.wikipedia.org/wiki/Semidefinite_programming) (SDP). These modules define the SDP.
* [solve_SDP.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/solve_SDP.py)/[solve_SDP_cg.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/solve_SDP_cg.py): These modules are used to solve the previously defined SDP.
* [calc_energies.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/calc_energies.py): Here, the lower bounds on the ground state energy are optimized based on different chosen parameters. The results are also saved.
* [main.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/main.py): Runs the optimization, once the paramerters are chosen accordingly.
* [analyze_data.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/analyze_data.py): Contains functions that extract and analyze the saved data.
* [plot_data.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/plot_data.py)/[thesis_plots.py](https://github.com/Daniel42Fan/masters_thesis/blob/main/thesis_plots.py): Contains plot routines to visualize the analyzed data for immediate investigation, as well as figures for the thesis itself. 
  
## Authors and acknowledgment
My master's thesis was written at the University of Vienna under the supervision of Univ.-Prof. Dr. Norbert Schuch.

Dr. Ilya Kull was my immediate supervisor.
