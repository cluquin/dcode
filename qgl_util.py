#!/usr/bin/python3
from math import log
from cmath import sqrt, exp, pi
import numpy as np
import scipy.sparse as sps
from functools import reduce
from itertools import permutations
from multiprocessing import Process, Pipe
import random
import re

# Global constants
# ================
# dictionary of local operators, local basis,
# and permutation lists for N2 and N3 ops
OPS = ({
    'I':np.array([[1.,0.],[0.,1.]]),
    'n':np.array([[0.,0.],[0.,1.]]),
    'nbar':np.array([[1.,0.],[0.,0.]]),
    'mix':np.array([[0.,1.],[1.,0.]]),
    'dead':np.array([[1.,0.]]).transpose(),
    'alive':np.array([[0.,1.]]).transpose(),
    'es'   : np.array([[1./sqrt(2), 1./sqrt(2)]]).transpose(),
    'permutations_3':list(set([perm for perm in
        permutations(['nbar','n','n','n'],4)])),
    'permutations_2':list(set([perm for perm in
        permutations(['nbar','nbar','n','n'],4)]))
})

# ops for trotter.py
ops = {
        'H' : 1.0 / sqrt(2.0) * \
              np.array( [[1.0,  1.0 ],[1.0,  -1.0]], dtype=complex),
        'I' : np.array( [[1.0,  0.0 ],[0.0,   1.0]], dtype=complex ),
        'X' : np.array( [[0.0,  1.0 ],[1.0,   0.0]], dtype=complex ),
        'Y' : np.array( [[0.0, -1.0j],[1.0j,  0.0]], dtype=complex ),
        'Z' : np.array( [[1.0,  0.0 ],[0.0 , -1.0]], dtype=complex ),
        'S' : np.array( [[1.0,  0.0 ],[0.0 , 1.0j]], dtype=complex ),
        'T' : np.array( [[1.0,  0.0 ],[0.0 , exp(1.0j*pi/4.0)]], dtype=complex ),
        '0' : np.array( [[1.0,   0.0],[0.0,   0.0]], dtype=complex ),
        '1' : np.array( [[0.0,   0.0],[0.0,   1.0]], dtype=complex ),
      }


# Matrix functions
# ================

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


# Kroeneker product list of matrices
# ----------------------------------
def matkron (matlist):
    return reduce(lambda A,B: np.kron(A,B),matlist)

# Kroeneker product list of sparse matrices
# -----------------------------------------
def spmatkron (matlist):
    return sps.csc_matrix(reduce(lambda A,B: sps.kron(A,B,'csc'),matlist))

# Hermitian conjugate
# -------------------
def dagger (mat):
    return mat.conj().transpose()


# Initial State Creation
# ======================
   
# Create Fock state
# -----------------
def fock (L, config, zero = 'dead', one = 'alive'):
    dec = int(config)
    state = [el.replace('0', zero).replace('1', one)
            for el in list('{0:0b}'.format(dec).rjust(L, '0'))]
    return matkron([OPS[key] for key in state])

# Random Fock state
# -----------------

def flip(p):
    return 1 if random.random() < p else 0

#http://stackoverflow.com/questions/12461361/bits-list-to-integer-in-python
def shifting(bitlist):
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out

def total_boolean(x):
    total = sum(x)
    if total == 2 or total == 3:
        return 1
    else:
        return 0

def rule(x, i):
    l = len(x)
    if i >=2 and i <= l-3:
        value = total_boolean([x[i-2], x[i-1], x[i+1], x[i+2]])
    elif i==0:
        value = total_boolean([x[i+1], x[i+2]])
    elif i==1:
        value = total_boolean([x[i-1], x[i+1], x[i+2]])
    elif i==l-1:
        value = total_boolean([x[i-2], x[i-1]])
    elif i==l-2:
        value = total_boolean([x[i-2], x[i-1], x[i+1]])        
    return value

def rule_boolean(x):
    alist = []
    for i in range(len(x)):
        alist.append(rule(x,i))
    return sum(alist) == 0

def test(L, p):
    bitlist = []
    while rule_boolean(bitlist):
        bitlist = []
        for i in range(L):
            bitlist.append(flip(p))
        print(bitlist)
    return bitlist

def randomfock(L, p):
    bitlist = []
    while rule_boolean(bitlist):
        bitlist = []
        for i in range(L):
            bitlist.append(flip(p))
    dec = str(shifting(bitlist))
    return 'd' + dec #fock(L, 'd'+dec)

# Create state with config - > binary: 0 - >dead, 1 -> 1/sqrt(2) (|0> +|1>)
# ------------------------------------------------------------------------
def local_superposition (L, config):
    return fock(L, config, one = 'es')

# Create state with one or two live sites
# ---------------------------------------
def one_alive (L, config):
    dec = 2**int(config)
    return fock(L, dec)

def two_alive(L, config):
    i, j = map(int, config.split('_'))
    return fock(L, 2**i + 2**j)

def two_es(L, config):
    i, j = map(int, config.split('_'))
    return local_superposition(L, 2**i + 2**j)

# Create state with all sites living
# ----------------------------------
def all_alive (L, config):
    dec = sum ([2**n for n in range(0,L)])
    return fock(L, dec)

# Create GHZ state
# ----------------
def GHZ (L, congif):
    s1=['alive']*(L)
    s2=['dead']*(L)
    return (1.0/sqrt(2.0)) \
            * ((matkron([OPS[key] for key in s1]) \
                + matkron([OPS[key] for key in s2])))

# Create W state
# --------------
def W (L, config):
    return (1.0/sqrt(L)) \
            * sum ([one_alive(L, k) for k in range(L)])

# Create as state with sites i and j maximally entangled
# reduces to 1/sqrt(2) (|01> - |10>) in L = 2 limit
# ------------------------------------------------------
def entangled_pair (L, config):
    i, j, k = map(int, config.split('_'))

    if k == 1:
        psi = 1./sqrt(2) * (fock(L, 2**j) + fock(L, 2**i))
    if k == 2:
        psi = 1./sqrt(2) * (fock(L, 2**j) - fock(L, 2**i))
    if k == 3:
        psi = 1./sqrt(2) * (fock(L, 0) + fock(L, 2**i + 2**j))
    if k == 4:
        psi = 1./sqrt(2) * (fock(L, 0) - fock(L, 2**i + 2**j))

    return psi

def singlet_array(L, config):
    singletlist = []
    singlet = (1/np.sqrt(2))*np.array([[0.0, 1.0, -1.0, 0.0]]).transpose()    
    if not L%2:
        for i in range(L//2):
            singletlist.append(singlet)
    if L%2:
        for i in range((L-1)//2):
            singletlist.append(singlet)
        singletlist.append(OPS['dead'])
    return matkron(singletlist)

def blinker_array(L, config):
    r = int(config)
    excitation_positions = np.array([0, 2] + [2 + n*(r+3) for n in range(1, ((L-3)//(r+3))+1)] + [2 + (n-1)*(r+3) + (r+1) for n in range(1, ((L-3)//(r+3)) + 1)])
    dec = np.sum(2**excitation_positions)
    return fock(L, dec)

def center(L, config):
    cpattern = re.compile('\d*[a-z]')
    dpattern = re.compile('[a-z]\d*')
    len_cent = int(re.search(cpattern, config).group()[:-1])    
    len_back = L - len_cent
    len_L = int(len_back/2)
    len_R = len_back - len_L
    cent_IC = [(re.search(dpattern, config).group(), 1)]
    left = fock(len_L, 0)
    cent = make_state(len_cent, cent_IC)
    right = fock(len_R, 0)
    if len_back == 0:
        return cent
    elif len_back == 1:
        return matkron([cent, right])
    else:
        return matkron([left, cent, right])

def random_state(L, config):
    np.random.seed(seed = int(config))
    state = np.random.normal(size = (2**L,1)) + 1.j*np.random.normal(size = (2**L,1))
    state = state/np.linalg.norm(state)
    return state


def basis_state(x):
    if x == 0.:
        return OPS['dead']
    if x == 1.:
        return OPS['alive']

def random_fock_state(L, config):
    np.random.seed(seed = int(config))
    state = np.zeros(L)
    state[np.random.random(size = L ) > 0.5] = 1
    print(state)
    state = [basis_state(x) for x in state]
    state = matkron(state)
    return state

def cluster_state(L, config):
    state = 1/np.sqrt(2**L)*np.ones((2**L,1))
    cphaseij = matkron([ops['1'], ops['S']]) + matkron([ops['0'], ops['I']])
    cphaseji = matkron([ops['S'], ops['1']]) + matkron([ops['I'], ops['0']])
    for i in range(L-1):
        state = op_on_state(np.dot(cphaseij, cphaseji), [i, i+1], state)
    state = state.reshape((2**L, 1))
    return state

# Make the specified state
# ------------------------

smap = { 'd' : fock,
         'l' : local_superposition,
         't' : two_es,
         'a' : all_alive,
         'c' : center,
         'G' : GHZ,
         'W' : W,
         'E' : entangled_pair,
         'S' : singlet_array,
         'B' : blinker_array,
         'r' : random_state,
         'C' : cluster_state,
         'R' : random_fock_state} 

def make_state (L, IC):
    state = np.asarray([[0.]*(2**L)]).transpose()
    for s in IC: 
            name = s[0][0]
            config = s[0][1:]
            coeff = s[1]
            state = state + coeff * smap[name](L, config)
    return state

if __name__ == "__main__":
    print(make_state(10,[('c3d5', 1.0)]).shape)
    print(make_state(10,[('r1', 1.0)]).shape)
    print(make_state(10,[('R3', 1.0)]).shape)
    print(make_state(10,[('C', 1.0)]).shape)
    # make_state(10,[('r1', 1.0)])
    # make_state(10,[('R1', 1.0)])
    # make_state(10,[('C', 1.0)])

