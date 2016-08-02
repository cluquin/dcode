#!/usr/bin/python3
import qgl
import qgl_util
import qgl_plotting
from math import sin, cos, pi
import numpy as np
#from mpi4py import MPI
import os
import sys
import random
sys.path.append(os.getcwd())
def function_creator(a):
    def linear_potential(x): return a*x
    return linear_potential

# QGL Simulation
# ==============

post_processing = True
gradient_simulation = False
restart = False

# Simulation Parameters
# ---------------------

L_list      = [10]# 12, 14, 16, 18, 20]
dt_list     = [0.1]
t_span_list = [(0, 100.)]
IC_list = [] 
output_dir  = 'qgol-exactrun115'
tasks       = ['t', 'n', 'nn', 'MI', 'localObs', 'EC-Center', 'SvN']
l = None

hamiltonian_types =['NN_0']#['NN_0','NN_1','NN_2','NN_01','NN_02','NN_12']
R_list = [12, 7, 2, 6, 14, 4, 21, 17, 10, 3, 23, 15, 28]
V_list = ['HP2_'+str(theta) for theta in [0,15,30,45,60,75,90]]
gradient_operator = '1'
gradients = []

if restart == True:
    lmax = 100
else:
    lmax = None

if gradient_simulation == True:
    for a in np.linspace(0, 1, 100):
        gradients.append([gradient_operator, function_creator(a), a])
else:
    gradients.append([gradient_operator, function_creator(0.0), 0.0])

# gradients = gradients[0:3]

def f(n):
    return 'c' + str(n) + 'd' + str(2**(n-1) + 2**(n-3) + 4 + 1) 

for L in L_list:
    IC_list.append([L, ('c3d5', 1.0)])
    IC_list.append([L, ('c2d3', 1.0)])
    IC_list.append([L, ('c5d27', 1.0)])
    IC_list.append([L, ('c6d51', 1.0)])
    IC_list.append([L, ('c7d99', 1.0)])
    IC_list.append([L, ('c8d195', 1.0)])
    IC_list.append([L, ('c3d7', 1.0)])
    IC_list.append([L, ('E' + str((L//2) - 1) + '_' + str((L//2)) + '_3', 1.0)])
    IC_list.append([L, ('E' + str((L//2) - 1) + '_' + str((L//2)) + '_4', 1.0)])
    IC_list.append([L, ('S', 1.0)])
    IC_list.append([L, ('r3', 1.0)])
    IC_list.append([L, ('R3', 1.0)])
    IC_list.append([L, ('C', 1.0)])

    # IC_list.append([L, ('G', 1.0)])
    # for n in range(9,17):
    #     IC_list.append([L, (f(n), 1.0)])    
    # IC_list.append([L, ('c8d165', 1.0)])
    # IC_list.append([L, ('B1', 1.0)])
    # IC_list.append([L, ('B2', 1.0)])
    # IC_list.append([L, ('E' + str((L//2)+1) + '_' + str((L//2) + 2) + '_3', 1.0)])
    # IC_list.append([L, ('E' + str((L//2)+1) + '_' + str((L//2) + 2) + '_4', 1.0)])
    # IC_list.append([L, ('c4d15', 1.0)])

# Simulations to Run
# ------------------
simulations = [ (tasks,  IC[0], t_span, dt, IC[1:], output_dir, gradient[0], gradient[1] , gradient[2], l, restart, lmax, hamiltonian_type)
                     for dt       in dt_list
                     for t_span   in t_span_list
                     for IC       in IC_list 
                     for gradient in gradients
                     for hamiltonian_type in hamiltonian_types
                     for V        in V_list]

# Run them
# --------
if not post_processing:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    for i, simulation in enumerate(simulations):
        if i % nprocs == rank:
            sim = qgl.Simulation (*simulation)
            del sim

# Post Processing
# ===============
if post_processing:
    qgl_plotting.main(output_dir, tasks, dt_list, \
                      t_span_list, IC_list, hamiltonian_types, V_list, gradients = gradients)
