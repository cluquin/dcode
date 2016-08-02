#!/usr/bin/python3

from multiprocessing import Pipe
from multiprocessing import Process

from os import makedirs, environ
from os.path import isfile
import time as lap 
import numpy as np
import scipy.sparse as sps
import scipy.io as sio
import scipy.linalg as sla
import scipy.sparse.linalg as spsla
from qgl_util import *
import measures as qms
from math import log
from cmath import sqrt, exp, pi
import cmath as cm
import time

class TrotterLayers:
    def __init__(self, L, R, geninteractions):
        self.ll = []
        self.lattice_length = L
        self.interaction_range = R
        self.interactions = []
        self.geninteractions = geninteractions

    def __iadd__(self, layer):
        self.ll.append(layer)
        return self

    def countlocal(self, i):
        count = 0
        for layer in self.ll:
            if layer.haslocal(i):
                count += 1
        return count

    def localweights(self):
        counts = {}
        for i in range(1, self.lattice_length + 1):
            counts[i] = 1./self.countlocal(i)
        return counts

    def countinteraction(self, i, j):
        count = 0
        for layer in self.ll:
            if layer.hasinteraction(i,j):
                count += 1
        return count

    def interaction(self):
        for layer in self.ll:
            for elem in layer.ll:
                templist = self.geninteractions(elem)
                
                for temp in templist:
                    self.interactions.append(temp)
        self.interactions = list(set(self.interactions))
        return self

    def interactionweights(self):
        counts = {}
        for interaction in self.interactions:
            counts[interaction] = 1./self.countinteraction(interaction[0], interaction[1])

        # for i in range(1, self.lattice_length-self.interaction_range+1):
        #     for j in range(i + 1,i + self.interaction_range + 1):
        #         counts[tuple([i,j])] = 1./self.countinteraction(i,j)
        return counts

class Layer():
    def __init__(self):
        self.ll = []

    def __iadd__(self, block):
        self.ll.append(block)
        return self
        
    def haslocal(self, i):
        for elem in self.ll:
            if i in elem:
                return True
        return False

    def hasinteraction(self, i, j):
        for elem in self.ll:
            if (i in elem) and (j in elem):
                return True
        return False

# ========================================
# Model class:
# computes/save Hamiltonian and Propagator
# time evolve an initial state
# note: Assumes dead BC's
# ========================================

def script_N(N, D=4, V='X'):
    perm = list(set([perm for perm in
        permutations(['0']*(D-N) + ['1']*N, D)]))
    script_N = np.zeros((2**(D+1), 2**(D+1)), dtype=complex)
    for tup in perm:
        matlist = [tup[i] for i in range(D//2)] + [ V ] + [tup[i] for i in range(D//2, D)]
        matlist = [ops[key] for key in matlist]
        script_N += matkron(matlist)
    return script_N

def op_on_state(meso_op, js, state, ds = None):
    if ds is None:
        L = int( log(len(state), 2) )
        ds = [2]*L
    else:
        L = len(ds)

    dn = np.prod(np.array(ds).take(js))
    dL = np.prod(ds)
    rest = np.setdiff1d(np.arange(L), js)
    ordering = list(rest) + list(js)

    new_state = state.reshape(ds).transpose(ordering)\
            .reshape(dL/dn, dn).dot(meso_op).reshape(ds)\
            .transpose(np.argsort(ordering)).reshape(dL)
    return new_state

def op_from_list(op_list_list):
    return  sum(matkron([ops[key] for key in op_list]) 
               for op_list in op_list_list)

def boundary_term_L(n , V = 'X'):
    if n == '0':
        return [[V, '0']]
    elif n == '1':
        return [[V, '1']]
    elif n == '2':
        return []

def boundary_term_L_N(n, V = 'X'):
    if n == '0':
        return [[V, '0', '0']]
    elif n == '1':
        return [[V, '0', '1'], [V, '1', '0']]
    elif n == '2':
        return [[V, '1', '1']]
    elif int(n) > 2:
        return []

def boundary_term_R_N(n, V = 'X'):
    if n == '0':
        return [['0', '0', V]]
    elif n == '1':
        return [['0', '1', V], ['1', '0', V]]
    elif n == '2':
        return [['1', '1', V]]
    elif int(n) > 2:
        return []

def boundary_term_l(n, V = 'X'):
    if n == '0':
        return [['0',V, '0', '0']]
    elif n == '1':
        return [['0',V, '0', '1'], ['0',V, '1', '0'], ['1',V,'0','0']]
    elif n == '2':
        return [['0',V, '1', '1'], ['1',V, '0', '1'], ['1',V,'1','0']]
    elif n == '3':
        return [['1',V, '1', '1']]
    elif int(n) > 3:
        return []

def boundary_term_r(n, V = 'X'):
    if n == '0':
        return [['0','0', V, '0']]
    elif n == '1':
        return [['0','0', V, '1'], ['0','1', V, '0'], ['1','0',V,'0']]
    elif n == '2':
        return [['0','1', V, '1'], ['1','0', V, '1'], ['1','1',V,'0']]
    elif n == '3':
        return [['1','1', V, '1']]
    elif int(n) > 3:
        return []

def boundary_term_R(n, V = 'X'):
    if n == '0':
        return [['0', V]]
    elif n == '1':
        return [['1', V]]
    elif n == '2':
        return []

        

def make_H_list(gradient_operator, gradient_function, L, hamiltonian_type = None, V = 'X'):
    if hamiltonian_type == None:
        L_list = [['X', '1', '1'], [gradient_operator, 'I', 'I']]
        l_list = [['0', 'X', '1', '1'],
                  ['1', 'X', '0', '1'],
                  ['1', 'X', '1', '0'],
                  ['1', 'X', '1', '1'],
                  ['I', gradient_operator, 'I', 'I']]

        r_list = [['1', '1', 'X', '0'],
                  ['1', '0', 'X', '1'],
                  ['0', '1', 'X', '1'],
                  ['1', '1', 'X', '1'],
                  ['I','I', gradient_operator, 'I']]
        R_list = [['1', '1', 'X'], ['I', 'I', gradient_operator]]

        HL = op_from_list(L_list[0:-1]) + gradient_function(0)*op_from_list(L_list[-1:])
        Hl = op_from_list(l_list[0:-1]) + gradient_function(1)*op_from_list(l_list[-1:])
        # Hj = script_N(2) + script_N(3)
        gradient_operator = op_from_list([['I', 'I', gradient_operator, 'I', 'I']])
        Hj = [script_N(2) + script_N(3) + gradient_function(j)*gradient_operator for j in range(2, L-2)]
        Hr = op_from_list(r_list[0:-1]) + gradient_function(L-2)*op_from_list(r_list[-1:])
        HR = op_from_list(R_list[0:-1]) + gradient_function(L-1)*op_from_list(R_list[-1:])
        H_list = [HL, Hl] + Hj + [Hr, HR]

    elif hamiltonian_type[0:2] == 'N_':
        if V[0:3] == 'HP_':
            ph = pi*float(V[3:])/180.0
            # ops['HP'] = np.dot(ops['H'], np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*ph)]], dtype=complex ))
            u1 = np.array([0.5*(1 + exp(1.0j*ph) - sqrt(1. + 6.*exp(1.0j*ph) + 1.*exp(2.0j*ph) )), 1], dtype = complex)
            u1 = u1 / np.linalg.norm(u1)
            u2 = np.array([0.5*(1 + exp(1.0j*ph) + sqrt(1. + 6.*exp(1.0j*ph) + 1.*exp(2.0j*ph) )), 1], dtype = complex)
            u2 = u2 / np.linalg.norm(u2)
            u = np.array([u1, u2], dtype = complex).transpose()
            udagger = u.conj().transpose()

            lambda1 = (1 - exp(1.0j*ph) - sqrt(1 + 6.*exp(1.0j*ph) + exp(2.0j*ph)))/(2.*sqrt(2.))
            lambda1p = 1.0j*cm.log(lambda1)
            lambda2 = (1 - exp(1.0j*ph) + sqrt(1 + 6.*exp(1.0j*ph) + exp(2.0j*ph)))/(2.*sqrt(2.))
            lambda2p = 1.0j*cm.log(lambda2)
            d = np.diag([lambda1p, lambda2p])

            ops['HP'] = np.dot(np.dot(u,d),udagger)

            
            L_list = []
            l_list = []
            R_list = []
            r_list = []
            bulk_operator = np.zeros((2**5, 2**5), dtype=complex)
            for elem in hamiltonian_type[2:]:
                L_list += boundary_term_L_N(elem, V = V[0:2])
                l_list += boundary_term_l(elem, V = V[0:2])
                R_list += boundary_term_R_N(elem, V = V[0:2])
                r_list += boundary_term_r(elem, V = V[0:2])
                bulk_operator += script_N(int(elem), V = V[0:2], D=4)
            L_list += [[gradient_operator, 'I', 'I']]
            l_list += [['I', gradient_operator, 'I', 'I']]
            R_list += [['I', 'I' , gradient_operator]]
            r_list += [['I', 'I', gradient_operator, 'I']]

            gradient_operator = op_from_list([['I','I', gradient_operator, 'I', 'I']])
            HL = op_from_list(L_list[0:-1]) + gradient_function(0)*op_from_list(L_list[-1:])            
            Hl = op_from_list(l_list[0:-1]) + gradient_function(1)*op_from_list(l_list[-1:])            
            HR = op_from_list(R_list[0:-1]) + gradient_function(L-1)*op_from_list(R_list[-1:])          
            Hr = op_from_list(r_list[0:-1]) + gradient_function(L-2)*op_from_list(r_list[-1:])                     
            Hj = [bulk_operator + gradient_function(j)*gradient_operator for j in range(2, L-2)]
            H_list = [HL] + [Hl] + Hj + [Hr] + [HR]
        elif V[0:4] == 'HP2_':
            ph = pi*float(V[4:])/180.0
            u = np.dot(ops['H'], np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*ph)]], dtype=complex ))
            ops['HP'] = u + u.conj().transpose()
            L_list = []
            l_list = []
            R_list = []
            r_list = []
            bulk_operator = np.zeros((2**5, 2**5), dtype=complex)
            for elem in hamiltonian_type[2:]:
                L_list += boundary_term_L_N(elem, V = V[0:2])
                l_list += boundary_term_l(elem, V = V[0:2])
                R_list += boundary_term_R_N(elem, V = V[0:2])
                r_list += boundary_term_r(elem, V = V[0:2])
                bulk_operator += script_N(int(elem), V = V[0:2], D=4)
            L_list += [[gradient_operator, 'I', 'I']]
            l_list += [['I', gradient_operator, 'I', 'I']]
            R_list += [['I', 'I' , gradient_operator]]
            r_list += [['I', 'I', gradient_operator, 'I']]

            gradient_operator = op_from_list([['I','I', gradient_operator, 'I', 'I']])
            HL = op_from_list(L_list[0:-1]) + gradient_function(0)*op_from_list(L_list[-1:])            
            Hl = op_from_list(l_list[0:-1]) + gradient_function(1)*op_from_list(l_list[-1:])            
            HR = op_from_list(R_list[0:-1]) + gradient_function(L-1)*op_from_list(R_list[-1:])          
            Hr = op_from_list(r_list[0:-1]) + gradient_function(L-2)*op_from_list(r_list[-1:])                     
            Hj = [bulk_operator + gradient_function(j)*gradient_operator for j in range(2, L-2)]
            H_list = [HL] + [Hl] + Hj + [Hr] + [HR]
        else:
            L_list = []
            l_list = []
            R_list = []
            r_list = []
            bulk_operator = np.zeros((2**5, 2**5), dtype=complex)
            for elem in hamiltonian_type[2:]:
                L_list += boundary_term_L_N(elem, V = V)
                l_list += boundary_term_l(elem, V = V)
                R_list += boundary_term_R_N(elem, V = V)
                r_list += boundary_term_r(elem, V = V)
                bulk_operator += script_N(int(elem), V = V, D=4)
            L_list += [[gradient_operator, 'I', 'I']]
            l_list += [['I', gradient_operator, 'I', 'I']]
            R_list += [['I', 'I' , gradient_operator]]
            r_list += [['I', 'I', gradient_operator, 'I']]

            gradient_operator = op_from_list([['I','I', gradient_operator, 'I', 'I']])
            HL = op_from_list(L_list[0:-1]) + gradient_function(0)*op_from_list(L_list[-1:])            
            Hl = op_from_list(l_list[0:-1]) + gradient_function(1)*op_from_list(l_list[-1:])            
            HR = op_from_list(R_list[0:-1]) + gradient_function(L-1)*op_from_list(R_list[-1:])          
            Hr = op_from_list(r_list[0:-1]) + gradient_function(L-2)*op_from_list(r_list[-1:])                     
            Hj = [bulk_operator + gradient_function(j)*gradient_operator for j in range(2, L-2)]
            H_list = [HL] + [Hl] + Hj + [Hr] + [HR]
    elif hamiltonian_type[0:2] == 'NN':
        if V[0:3] == 'HP_':
            ph = pi*float(V[3:])/180.0
            # ops['HP'] = np.dot(ops['H'], np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*ph)]], dtype=complex ))
            u1 = np.array([0.5*(1 + exp(1.0j*ph) - sqrt(1. + 6.*exp(1.0j*ph) + 1.*exp(2.0j*ph) )), 1], dtype = complex)
            u1 = u1 / np.linalg.norm(u1)
            u2 = np.array([0.5*(1 + exp(1.0j*ph) + sqrt(1. + 6.*exp(1.0j*ph) + 1.*exp(2.0j*ph) )), 1], dtype = complex)
            u2 = u2 / np.linalg.norm(u2)
            u = np.array([u1, u2], dtype = complex).transpose()
            udagger = u.conj().transpose()

            lambda1 = (1 - exp(1.0j*ph) - sqrt(1 + 6.*exp(1.0j*ph) + exp(2.0j*ph)))/(2.*sqrt(2.))
            lambda1p = 1.0j*cm.log(lambda1)
            lambda2 = (1 - exp(1.0j*ph) + sqrt(1 + 6.*exp(1.0j*ph) + exp(2.0j*ph)))/(2.*sqrt(2.))
            lambda2p = 1.0j*cm.log(lambda2)
            d = np.diag([lambda1p, lambda2p])

            ops['HP'] = np.dot(np.dot(u,d),udagger)
            
            L_list = []
            R_list = []
            bulk_operator = np.zeros((2**3, 2**3), dtype=complex)
            for elem in hamiltonian_type[3:]:
                L_list += boundary_term_L(elem, V = V[0:2])
                R_list += boundary_term_R(elem, V = V[0:2])
                bulk_operator += script_N(int(elem), V = V[0:2], D=2)
            L_list += [[gradient_operator, 'I']]
            R_list += [['I', gradient_operator]]
            gradient_operator = op_from_list([['I', gradient_operator, 'I']])
            HL = op_from_list(L_list[0:-1]) + gradient_function(0)*op_from_list(L_list[-1:])            
            HR = op_from_list(R_list[0:-1]) + gradient_function(L-1)*op_from_list(R_list[-1:])                   
            Hj = [bulk_operator + gradient_function(j)*gradient_operator for j in range(1, L-1)]
            H_list = [HL] + Hj + [HR]
        elif V[0:4] == 'HP2_':
            ph = pi*float(V[4:])/180.0
            u = np.dot(ops['H'], np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*ph)]], dtype=complex ))
            ops['HP'] = u + u.conj().transpose()
            L_list = []
            R_list = []
            bulk_operator = np.zeros((2**3, 2**3), dtype=complex)
            for elem in hamiltonian_type[3:]:
                L_list += boundary_term_L(elem, V = V[0:2])
                R_list += boundary_term_R(elem, V = V[0:2])
                bulk_operator += script_N(int(elem), V = V[0:2], D=2)
            L_list += [[gradient_operator, 'I']]
            R_list += [['I', gradient_operator]]
            gradient_operator = op_from_list([['I', gradient_operator, 'I']])
            HL = op_from_list(L_list[0:-1]) + gradient_function(0)*op_from_list(L_list[-1:])            
            HR = op_from_list(R_list[0:-1]) + gradient_function(L-1)*op_from_list(R_list[-1:])                   
            Hj = [bulk_operator + gradient_function(j)*gradient_operator for j in range(1, L-1)]
            H_list = [HL] + Hj + [HR]
        else:
            L_list = []
            R_list = []
            bulk_operator = np.zeros((2**3, 2**3), dtype=complex)
            for elem in hamiltonian_type[3:]:
                L_list += boundary_term_L(elem, V = V)
                R_list += boundary_term_R(elem, V = V)
                bulk_operator += script_N(int(elem), V = V, D=2)
            L_list += [[gradient_operator, 'I']]
            R_list += [['I', gradient_operator]]
            gradient_operator = op_from_list([['I', gradient_operator, 'I']])
            HL = op_from_list(L_list[0:-1]) + gradient_function(0)*op_from_list(L_list[-1:])            
            HR = op_from_list(R_list[0:-1]) + gradient_function(L-1)*op_from_list(R_list[-1:])                   
            Hj = [bulk_operator + gradient_function(j)*gradient_operator for j in range(1, L-1)]
            H_list = [HL] + Hj + [HR]
    return H_list

# def make_H_list():
#     L_list = [['X', '1', '1']]
#     l_list = [['0', 'X', '1', '1'],
#               ['1', 'X', '0', '1'],
#               ['1', 'X', '1', '0'],
#               ['1', 'X', '1', '1']]

#     r_list = [['1', '1', 'X', '0'],
#               ['1', '0', 'X', '1'],
#               ['0', '1', 'X', '1'],
#               ['1', '1', 'X', '1']]
#     R_list = [['1', '1', 'X']]

#     HL = op_from_list(L_list)
#     Hl = op_from_list(l_list)
#     Hj = script_N(2) + script_N(3)
#     Hr = op_from_list(r_list)
#     HR = op_from_list(R_list)
#     H_list = [HL, Hl, Hj, Hr, HR]
#     return H_list

def make_H_list_spin_res(R):

    L_list = [['X', '1', '1']]
    l_list = [['0', 'X', '1', '1'],
              ['1', 'X', '0', '1'],
              ['1', 'X', '1', '0'],
              ['1', 'X', '1', '1']]

    r_list = [['1', '1', 'X', '0'],
              ['1', '0', 'X', '1'],
              ['0', '1', 'X', '1'],
              ['1', '1', 'X', '1']]
    R_list = [['1', '1', 'X']]
    
    Hsigmax_list = []# [['X']+['I']*R]
    for i in range(R):
        Hsigmax_list.append(['I']*i+['X']+['I']*(R-i))
    print(Hsigmax_list)
    Hsigmaz_list = [['Z']+['I']*R]
    Hinteraction_list = []
    for i in range(R):
        Hinteraction_list.append(['Z']+['I']*i+['Z']+['I']*(R-1-i))
    Hsigmax = op_from_list(Hsigmax_list)
    Hsigmaz = op_from_list(Hsigmaz_list)
    Hinteraction = op_from_list(Hinteraction_list)
    H = Hsigmax + Hsigmaz + Hinteraction
    return H

def H_to_U(H, dt):
    return sla.expm(-1j*H*dt)

# def make_U_dict(H_list, dt, U_keys = None):
#     if U_keys == None:
#         U_keys = ['L'+str(dt), 'l'+str(dt), 'j'+str(dt), 
#                   'r'+str(dt), 'R'+str(dt)]
#     U_list = [H_to_U(H, dt) for H in H_list]
#     U_dict = dict(zip(U_keys, U_list))
#     return U_dict

def make_U_dict(H_list, dt, U_keys = None):
    if U_keys == None:
        # U_keys = ['L'+str(dt), 'l'+str(dt), 'j'+str(dt), 
        #           'r'+str(dt), 'R'+str(dt)]
        U_keys = ['L'+str(dt), 'l'+str(dt)] + [str(j)+str(dt) for j in range(2, len(H_list)-2)] + ['r'+str(dt), 'R'+str(dt)]
    U_list = [H_to_U(H, dt) for H in H_list]
    U_dict = dict(zip(U_keys, U_list))
    return U_dict

def get_U_js_pair(U_dict, j, L, dt, hamiltonian_type = None):
    if hamiltonian_type == None:
        if j == 0:
            U = U_dict['L'+str(dt)]
            js = [0, 1, 2]
        elif j == 1:
            U = U_dict['l'+str(dt)]
            js = [0, 1, 2, 3]
        elif j == L-2:
            U = U_dict['r'+str(dt)]
            js = [L-4, L-3, L-2, L-1]
        elif j == L-1:
            U = U_dict['R'+str(dt)]
            js = [L-3, L-2, L-1]
        else:
            U = U_dict[str(j)+str(dt)]
            js = [j-2, j-1, j, j+1, j+2]
    elif hamiltonian_type[0:2] == 'N_':
        if j == 0:
            U = U_dict['L'+str(dt)]
            js = [0, 1, 2]
        elif j == 1:
            U = U_dict['l'+str(dt)]
            js = [0, 1, 2, 3]
        elif j == L-2:
            U = U_dict['r'+str(dt)]
            js = [L-4, L-3, L-2, L-1]
        elif j == L-1:
            U = U_dict['R'+str(dt)]
            js = [L-3, L-2, L-1]
        else:
            U = U_dict[str(j)+str(dt)]
            js = [j-2, j-1, j, j+1, j+2]        
    elif hamiltonian_type[0:2] == 'NN':
        if j == 0:
            U = U_dict['L'+str(dt)]
            js = [0, 1]
        elif j == L-1:
            U = U_dict['R'+str(dt)]
            js = [L-2, L-1]
        else:
            U = U_dict[str(j)+str(dt)]
            js = [j-1, j, j+1]        
    return U, js

# def get_U_js_pair(U_dict, j, L, dt):
#     if j == 0:
#         U = U_dict['L'+str(dt)]
#         js = [0, 1, 2]
#     elif j == 1:
#         U = U_dict['l'+str(dt)]
#         js = [0, 1, 2, 3]
#     elif j == L-2:
#         U = U_dict['r'+str(dt)]
#         js = [L-4, L-3, L-2, L-1]
#     elif j == L-1:
#         U = U_dict['R'+str(dt)]
#         js = [L-3, L-2, L-1]
#     else:
#         U = U_dict['j'+str(dt)]
#         js = [j-2, j-1, j, j+1, j+2]
#     return U, js

def trotter_layer(n, dt, L, state, U_dict, hamiltonian_type = None, R = 5):
    for j in range(n, L, R):
        U, js = get_U_js_pair(U_dict, j, L, dt, hamiltonian_type = hamiltonian_type)
        state = op_on_state(U, js, state)
    return state

def interactions(block):
    L = len(block)
    interactions = []
    for i in range(L):
        for j in range(i+1, L):
            interactions.append(tuple([block[i], block[j]]))
    return interactions

def spin_res_layers(L, R):
    layers = TrotterLayers(L, R, interactions)#[]
    for i in range(1, R+2):
        layer = Layer()
        for j in range(i, L, R+1):
            if j+R <= L:
                layer += tuple(range(j, j+R+1))
        layers += layer
    return layers

def spin_res_op(interactions, interactionweights, local, localweights, Omega, delta0, J):
    left = np.min(interactions)
    right = np.max(interactions)
    L = right-left+1
    interactionoperator = np.zeros((2**L, 2**L), dtype = complex) 
    localoperator = np.zeros((2**L, 2**L), dtype = complex) 

    for interaction in interactions:
        operators  = ['I']*L
        weight = -1.*J*interactionweights[interaction]
        operators[interaction[0]-left] = 'Z'
        operators[interaction[1]-left] = 'Z'
        interactionoperator += weight*op_from_list([operators])

    for localterm in local:
        sigmaz_operators = ['I']*L
        sigmax_operators = ['I']*L
        weight = localweights[localterm]
        sigmaz_operators[localterm-left] = 'Z'
        sigmax_operators[localterm-left] = 'X'
        localoperator += delta0*weight*op_from_list([sigmaz_operators])
        localoperator += Omega*weight*op_from_list([sigmax_operators])
    operator = localoperator + interactionoperator
    return operator

def spin_res_ops(L, R, Omega, delta0, J):
    layers = spin_res_layers(L, R)
    layers.interaction()
    interactionweights = layers.interactionweights()
    localweights = layers.localweights()
    operator_list = []
    for layer in layers.ll:
        operators = []
        for elem in layer.ll:
            interactionterm = interactions(elem)
            operator = spin_res_op(interactionterm, interactionweights, elem, 
                                   localweights, Omega, delta0, J)
            operators.append(operator)
        operator_list.append(operators)
    return layers, operator_list

def build_spin_res_propagators(operator_list, dt):
    propagator_list = []
    nlayers = len(operator_list)
    n = 0
    for layer in operator_list:
        operators = []
        if n == nlayers:
            for operator in layer:
                operators.append(H_to_U(operator, dt))
        else:
            for operator in layer:
                operators.append(H_to_U(operator, 0.5*dt))
        propagator_list.append(operators)
    return propagator_list

# layers, operators = spin_res_ops(10, 2, 1.0, 1.0, 1.0)
# layers = np.asarray([np.array(layer.ll) for layer in layers.ll])
# propagators = build_spin_res_propagators(operators, 0.1)

class Model:
    # Build a model
    # -------------
    def __init__ (self, L, dt, IC,
                    model_dir = environ['HOME']+'/Documents/qgl_exact/'):
        self.L  = L
        self.dt = dt

        self.curr_state = IC
        self.state_list = []
        self.measures   = []

        # self.ham_name = 'L'+str(self.L)+'_qgl_ham.mtx'
        # self.prop_name = 'L{}_dt{}'.format(self.L, self.dt)+'_qgl_prop'
        # self.ham_path  = model_dir + 'hamiltonians/'+self.ham_name
        # self.prop_path = model_dir + 'propagators/'+self.prop_name
        return

    # totalistic selector/swap for 3 live sites 
    # -----------------------------------------
    # def N3 (self,k):
    #     n3=0
    #     for  tup in OPS['permutations_3']:
    #         local_matlist3 = [tup[0],tup[1],'mix',tup[2],tup[3]]

    #         if k==0:
    #             del local_matlist3[0]
    #             del local_matlist3[0]
    #         if k==self.L-1:
    #             del local_matlist3[-1]
    #             del local_matlist3[-1]

    #         if k==1:
    #             del local_matlist3[0]
    #         if k==self.L-2:
    #             del local_matlist3[-1]

    #         matlist3 = ['I']*(k-2)+local_matlist3
    #         matlist3 = matlist3 +['I']*(self.L-len(matlist3))
    #         matlist3 = [OPS[key] for key in matlist3]
    #         n3 = n3 + spmatkron(matlist3)
    #     return n3

    # totalistic selector/swap for 2 live sites 
    # -----------------------------------------
    # def N2(self,k):
    #     n2 = 0
    #     for tup in OPS['permutations_2']:
    #         local_matlist2 = [tup[0],tup[1],'mix',tup[2],tup[3]]
    #         if k==0:
    #             del local_matlist2[0]
    #             del local_matlist2[0]
    #         if k==self.L-1:
    #             del local_matlist2[-1]
    #             del local_matlist2[-1]

    #         if k==1:
    #             del local_matlist2[0]
    #         if k==self.L-2:
    #             del local_matlist2[-1]

    #         matlist2 = ['I']*(k-2)+local_matlist2
    #         matlist2 = matlist2+['I']*(self.L-len(matlist2))
    #         matlist2 = [OPS[key] for key in matlist2]
    #         n2 = n2 + spmatkron(matlist2)
    #     return n2


    # def boundary_terms_gen(self, L):
    #     L_terms = [
    #               ['mix',  'n',   'n',    'I'   ] + ['I']*(L-4),
    #               ['nbar', 'mix', 'n',    'n'   ] + ['I']*(L-4),
    #               ['n',    'mix', 'nbar', 'n'   ] + ['I']*(L-4),
    #               ['n',    'mix', 'n',    'nbar'] + ['I']*(L-4),
    #               ['n',    'mix', 'n',    'n'   ] + ['I']*(L-4)
    #               ] 

    #     R_terms = [
    #               ['I']*(L-4) + ['n',    'n',    'mix', 'nbar'],
    #               ['I']*(L-4) + ['n',    'nbar', 'mix', 'n'  ],
    #               ['I']*(L-4) + ['nbar', 'n',    'mix', 'n'  ],
    #               ['I']*(L-4) + ['n',    'n',    'mix', 'n'  ],
    #               ['I']*(L-4) + ['I',    'n',    'n',   'mix']
    #               ]

    #     boundary_terms = L_terms + R_terms
    #     return boundary_terms

    # Create the Hamiltonian and propagator
    # ------------------------------------- 
    # def gen_model (self):
    #     # Hamiltonian
    #     if isfile(self.ham_path):
    #         print('Importing Hamiltonian...')
    #         H = sio.mmread(self.ham_path).tocsc()
    #     else:
    #         print('Building Hamiltonian...')
    #         H = sum ([(self.N2(k) + self.N3(k)) for k in range(2, self.L-2)])
    #         for matlistb in self.boundary_terms_gen(self.L):
    #             matlistb = [OPS[key] for key in matlistb]
    #             H = H + spmatkron(matlistb)
    #     self.ham = H

        # # Propagator
        # if isfile(self.prop_path):
        #     print('Importing propagator...')
        #     U0 = np.fromfile (self.prop_path)
        #     U_dim = 2**(self.L)
        #     U0 = ( U0[0:len(U0)-1:2] + 1j*U0[1:len(U0):2] ).reshape((U_dim,U_dim))
        # else:
        #     print('Building propagator...')
        #     U0 = spsla.expm(-1j*self.dt*H).todense()
        # self.prop = np.asarray(U0)

    # Save the Hamiltonian (sparse) and propagator (dense)
    # ----------------------------------------------------
    def write_out (self):
        sio.mmwrite(self.ham_path, self.ham)
        self.prop.tofile(self.prop_path)

    # Generate states up to nmax
    # --------------------------
    def time_evolve (self, nmax):
        new_states = [self.curr_state] * (nmax-len(self.state_list))
        print('Time evolving IC...')
        for i in range(1,len(new_states)):
            new_states[i] = self.prop.dot(new_states[i-1])

        self.state_list += new_states
        self.curr_state = self.state_list[-1]



    # def trotter_sym(M, dt, state, U_dict):
    #     #Time steps
    #     # dts = [dt/4, dt/2, dt/4, dt/2, dt/2, dt, dt/2, dt/2, dt/4, dt/2, dt/4]
    #     dts = [dt/2, dt/2, dt/2, dt/2, dt, dt/2, dt/2, dt/2, dt/2]
    #     # dts = [dt/8, dt/4, dt/8, dt/2, dt/8, dt/4, dt/8, dt/2, dt, dt/2, dt/8, dt/4,
    #     #        dt/8, dt/2, dt/8, dt/4, dt/8]
    #     #Series of propagators to apply
    #     # ns = [0, 1, 0, 2, 3, 4, 3, 2, 0, 1, 0]
    #     ns = [0, 1, 2, 3, 4, 3, 2, 1, 0]
    #     # ns = [4, 3, 4, 2, 4, 3, 4, 1, 0, 1, 4, 3, 4, 2, 4, 3, 4]
    #     state = state
    #     yield state
    #     #Number of time steps
    #     for m in range(1,M):
    #         #Application of propagators to state for m'th time step
    #         for n, dt in zip(ns, dts):
    #             state = trotter_layer(n, dt, L, state, U_dict)
    #         #Return state at each time step
    #         yield state

    # def trotter_asym(M, dt, state, U_dict):
    #     #Time steps
    #     dts = [dt]*5
    #     state = state
    #     yield state
    #     #Number of time steps
    #     for m in range(1,M):
    #         for n, dt in zip(range(5), dts):
    #             #Application of propagators to state for m'th time step
    #             state = trotter_layer(n, dt, L, state, U_dict)
    #         #Return state at each time step
    #         yield state

    def trotter_evolve(self, dt, nmax, meas, gradient_operator, gradient_function, t0 = 0, l = None,
                       sim_name = None, restart = False, lmax = None, hamiltonian_type = None, V = 'X'):
        # new_states =  [self.curr_state] * (nmax - len(self.state_list))
        dt = dt
        M = nmax
        if restart == True:
            meas.measures = qms.Measurements.read_in(0, meas.meas_file)
        else:
            meas = meas
        meas = meas
        dim = len(self.curr_state)
        
        if not hamiltonian_type == None:
            if hamiltonian_type[0:2] == 'NN':
                ns = [0, 1, 2, 1, 0]
                dts = [dt/2, dt/2, dt, dt/2, dt/2]
                R = 3
            else:
                R = 5
                ns = [0, 1, 2, 3, 4, 3, 2, 1, 0]
                dts = [dt/2, dt/2, dt/2, dt/2, dt, dt/2, dt/2, dt/2, dt/2]

        else:
            R = 5
            ns = [0, 1, 2, 3, 4, 3, 2, 1, 0]
            dts = [dt/2, dt/2, dt/2, dt/2, dt, dt/2, dt/2, dt/2, dt/2]


        print('Time evolving IC...')
        H_list = make_H_list(gradient_operator, gradient_function, self.L, hamiltonian_type = hamiltonian_type, V = V)

        
        if not hamiltonian_type == None:
            if hamiltonian_type[0:2] == 'NN':
                U_dict = make_U_dict(H_list, dt, U_keys = ['L'+str(dt)] + [str(j)+str(dt) for j in range(1, len(H_list)-1)] + ['R'+str(dt)])
                U_dict.update(make_U_dict(H_list, dt/2, U_keys = ['L'+str(dt/2)] + [str(j)+str(dt/2) for j in range(1, len(H_list)-1)] + ['R'+str(dt/2)]))
            else:
                U_dict = make_U_dict(H_list, dt)
                U_dict.update(make_U_dict(H_list, dt/2, U_keys = ['L'+str(dt/2), 'l'+str(dt/2)] + [str(j)+str(dt/2) for j in range(2, len(H_list)-2)] + ['r'+str(dt/2), 'R'+str(dt/2)]))
        else:
            U_dict = make_U_dict(H_list, dt)
            U_dict.update(make_U_dict(H_list, dt/2, U_keys = ['L'+str(dt/2), 'l'+str(dt/2)] + [str(j)+str(dt/2) for j in range(2, len(H_list)-2)] + ['r'+str(dt/2), 'R'+str(dt/2)]))


        if l == None:
            meas.measure(self.curr_state, t0)

            for m in range(1, M+1):
                #Application of propagators to state for m'th time step
                for n, dt0 in zip(ns, dts):
                    self.curr_state = trotter_layer(n, dt0, self.L, self.curr_state, U_dict, hamiltonian_type = hamiltonian_type, R = R)
                self.curr_state = self.curr_state.reshape((dim, 1))
                #Return state at each time step
                meas.measure(self.curr_state, t0 + m*dt)
            meas.write_out()
        else:
            if restart == True:
                self.curr_state = np.loadtxt(meas.meas_file[:-5]+'_t{}'.format(lmax)+'.dat').view(complex)
                for m in range(lmax+1, M+1):
                    #Application of propagators to state for m'th time step
                    for n, dt0 in zip(ns, dts):
                        self.curr_state = trotter_layer(n, dt0, self.L, self.curr_state, U_dict, hamiltonian_type = hamiltonian_type, R = R)

                    self.curr_state = self.curr_state.reshape((dim, 1))
                    #Return state at each time step
                    meas.measure(self.curr_state, t0 + m*dt)
                    if not m % l:
                        np.savetxt(meas.meas_file[:-5]+'_t{}'.format(m)+'.dat', self.curr_state.view(float))
                        meas.write_out()
            else:
                meas.measure(self.curr_state, t0)
                for m in range(1, M+1):
                    #Application of propagators to state for m'th time step
                    for n, dt0 in zip(ns, dts):
                        self.curr_state = trotter_layer(n, dt0, self.L, self.curr_state, U_dict, hamiltonian_type = hamiltonian_type, R = R)
                    self.curr_state = self.curr_state.reshape((dim, 1))
                    #Return state at each time step
                    meas.measure(self.curr_state, t0 + m*dt)
                    if not m % l:
                        np.savetxt(meas.meas_file[:-5]+'_t{}'.format(m)+'.dat', self.curr_state.view(float))
                        meas.write_out()
                    
    def spin_res_trotter_evolve(self, L, R, Omega, delta0, J, dt, nmax, meas, t0 = 0):
        dt = dt
        M = nmax
        meas = meas
        dim = len(self.curr_state)
        meas.measure(self.curr_state, t0)
        layers, operators = spin_res_ops(L, R, Omega, delta0, J)
        layers = np.asarray([np.array(layer.ll) for layer in layers.ll])-1
        propagators = build_spin_res_propagators(operators, dt)
        ns = range(R+1) + list(np.abs(R-1-np.range(R)))
        
        for m in range(1, M):
            for n in ns:
                for i in range(len(layers[n])):
                    self.curr_state = op_on_state(propagators[n][i], list(layers[n][i]), 
                                                  self.curr_state)
            meas.measure(self.curr_state, t0 + m*dt)
        meas.write_out()

#==========================================================================
# Simulation class
# Model instance and Measurements instance assined to a Simulation instance
#==========================================================================

class Simulation():
    def __init__ (self, tasks, L, t_span, dt, IC, output_dir, gradient_operator,
                  gradient_function, a, l, restart, lmax, hamiltonian_type, V, model_dir = environ['HOME']+'/Documents/qgl_exact/', seed = "March 30, 2016 3:28 PM"):

        # makedirs('/media/dvargas/mypassport/qgol-exact/'+output_dir, exist_ok=True)
        makedirs(environ['HOME'] + '/scratch/' + output_dir, exist_ok=True)
        self.tasks = tasks
        self.L = L
        self.dt = dt
        self.nmin = round(t_span[0]/self.dt)
        self.nmax = round(t_span[1]/self.dt)

        self.gradient_operator = gradient_operator
        self.gradient_function = gradient_function
        self.a  = a
        self.l = l
        self.restart = restart
        self.lmax = lmax
        self.hamiltonian_type = hamiltonian_type
        self.V = V

        random.seed(seed)
        self.IC_vec = make_state(self.L, IC)
        IC_name = '-'.join(['{}{:0.3f}'.format(name, val) \
                for (name, val) in IC])
        if self.hamiltonian_type == None:
            self.sim_name = ('L{}_dt{}_t_span{}-{}_IC{}_a{}'+'_V'+V).format ( \
                L, dt, t_span[0], t_span[1], IC_name, a)
        else:
            self.sim_name = ('L{}_dt{}_t_span{}-{}_IC{}_a{}'+hamiltonian_type+'_V'+ V).format ( \
                L, dt, t_span[0], t_span[1], IC_name, a)
        # meas_file = '/media/dvargas/mypassport/qgol-exact/'+output_dir+'/'+self.sim_name+'.meas'
        meas_file = environ['HOME']+'/scratch/'+output_dir+'/'+self.sim_name+'.meas'
        self.model = Model (L, dt, self.IC_vec, model_dir = model_dir)
        self.meas = qms.Measurements (tasks = tasks, meas_file = meas_file)

        self.run_sim()
        return

    # model -> states -> measurements then save
    def run_sim (self):
        # self.model.gen_model()
        # self.model.time_evolve(self.nmax)
        t0 = time.time()
        # self.model.trotter_evolve(self.dt, self.nmax, self.meas)
        self.model.trotter_evolve(self.dt, self.nmax, self.meas, self.gradient_operator, self.gradient_function, l = self.l, sim_name = self.sim_name, restart = self.restart, lmax = self.lmax, hamiltonian_type = self.hamiltonian_type, V = self.V)
        t1 = time.time()
        print('L = ', self.L, t1-t0)
        # self.model.write_out ()
        # self.meas.take_measurements (self.model.state_list, \
        #         self.dt, self.nmin, self.nmax)
        # self.meas.write_out ()
