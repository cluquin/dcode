#!/usr/bin/python3
import numpy as np
import networkx as nx
import itertools
import scipy.stats as stats
import essentials as es
import random
import time
#Scripted by David Vargas
#---------------------------------------------
#Convention if directed networks are introduced,
#Aij=1 if there is an edge from j to i. This
#means one would sum over row i to get the total
#number of incoming edges to node i.
#Here I am summing over rows.
#Unfinished function - quantum assortativity
#Next implement hierarchical clustering algorithm
#described on page 388 of Newman's book on networks
#Next implement fractal dimension calculation proposed
#by Simona.

def strengths(mutualinformation):
    #Return the strength of all nodes in the network in order.
    return np.sum(mutualinformation, axis = 1)

def density(matrix):
    #Calculates density, also termed connectance in some
    #literature. Defined on page 134 of Mark Newman's book
    #on networks.
    l = len(matrix)
    lsq = l*(l-1)
    return np.sum(matrix)/lsq 

# def clustering(matrix):
#     #Calculates the clustering coefficient
#     #as it is defined in equation 7.39 of
#     #Mark Newman's book on networks. Page 199.
#     matrixcube = np.linalg.matrix_power(matrix, 3)
#     matrixsq = np.linalg.matrix_power(matrix, 2)
#     #Zero out diagonal entries. So we do not count edges as
#     #connected triples.
#     for i in range(len(matrixsq)):
#         matrixsq[i][i] = 0
#     denominator = np.sum(matrixsq)
#     numerator = np.trace(matrixcube)
#     #if there are no closed paths of length
#     #three the clustering is automatically
#     #set to zero.
#     if numerator == 0.:
#         return 0.
#     else:
#         return numerator/denominator

def clustering(matrix):
    #Calculates the clustering coefficient
    #as it is defined in equation 7.39 of
    #Mark Newman's book on networks. Page 199.
    matrixcube = np.linalg.matrix_power(matrix, 3)
    numerator = np.trace(matrixcube)
    #if there are no closed paths of length
    #three the clustering is automatically
    #set to zero.
    if numerator == 0.:
        return 0.
    else:
        matrixsq = np.linalg.matrix_power(matrix, 2)
        denominator = np.sum(matrixsq) - np.trace(matrixsq)
        return numerator/denominator

def localclustering(matrix):
    l = len(matrix)
    localclustering = []
    matrixcube = np.linalg.matrix_power(matrix, 3)
    for i in range(l):
        matrix2 = np.outer(matrix[i,:], matrix[i,:])
        numerator = matrixcube[i][i]
        denominator = (np.sum(matrix2)-np.trace(matrix2))
        if denominator != 0:
            localclustering.append(numerator/denominator)
        else:
            localclustering.append(0)
    # squaretmp = 0
    # for i in range(l):
    #     for j in range(l):
    #         for k in range(j):
    #             squaretmp += matrix[i][j]*matrix[i][k]
    #     if squaretmp != 0:
    #         localclustering.append(0.5*matrixcube[i][i]/squaretmp)
    #     else:
    #         localclustering.append(0)
        # squaretmp = 0
    return np.array(localclustering)

def rhotype(rho):
    localtmp = 0
    localtmp2 = 0
    d = len(rho)
    basisstate = np.zeros(d)
    brastate = np.zeros(d)
    #By construction I do not have to conjugate the elements to get the bra.
    #This is because basis state is a vector with a single one in a list of zeros.
    for i in range(d):
        basisstate[i] = 1
        brastate = basisstate
        #Map basis state to basis state.
        basisstate = np.dot(rho, basisstate)
#        print 'basis',basisstate
#        print 'bra',brastate,'basis',basisstate
#        print 'rho',rho
        expectation=np.dot(brastate, basisstate)
        basisstate[i] = 0
#        print tuple([expectation,i])
        if expectation >= localtmp:
            localtmp = expectation
            localtmp2 = i
    rhotype = localtmp2
    return int(rhotype)

def rhotypes(rhos):
    L = len(rhos)
    rhoTypes = np.zeros(L, dtype = int)
    for i in range(L):
        rhoTypes[i] = rhotype(rhos[i])
    return rhoTypes

def quantumassortativity(rhos,mutualinformation,typefunction):
    L = len(rhos)
    sites = np.array(range(L))
    rhoTypes = typefunction(rhos)
    types = np.unique(rhoTypes)
    longtype = {}
    for atype in types:
        longtype[atype] = atype*np.ones(L, dtype = int)
    ed = len(types)
    eab = np.zeros((ed,ed))
    sitesbytype = {}
    sitesbytype2 = []
    for atype in types:
        abool = np.equal(rhoTypes,longtype[atype])
        sitesbytype[atype] = list(sites[abool])
    for atype in types:
        for btype in types:
            for i in sitesbytype[atype]:
                for j in sitesbytype[btype]:
                    eab[atype][btype] += mutualinformation[i][j]
    eab = eab/np.sum(eab)
    trace = np.trace(eab)
    prodsum = np.sum(np.dot(eab,eab))
    if 1. - prodsum != 0:
        assortativity = (trace-prodsum)/(1.-prodsum)
        return assortativity
    else:
        return nan
  
# def disparity(matrix):
#     #Disparity defined on page 199 of doi:10.1016/j.physrep.2005.10.009 
#     #Equation 2.39, Here I take the average of this quantity over the
#     #entire network
#     l = len(matrix)
#     numerator = np.sum(matrix**2, axis = 1)/l
#     denominator = np.sum(matrix, axis = 1)
#     #Logical Check
#     logos = denominator > 0
#     numerator = numerator[logos]
#     denominator = denominator[logos]
#     denominator = denominator**2
#     #Check for zero denominator.

#     if sum(denominator) == 0.:
#         return np.nan
#     else: 
#         return sum(numerator/denominator)

def disparity(matrix):
    #Disparity defined on page 199 of doi:10.1016/j.physrep.2005.10.009 
    #Equation 2.39, Here I take the average of this quantity over the
    #entire network
    l = len(matrix)
    numerator = np.sum(matrix**2, axis = 1)/l
    denominator = np.sum(matrix, axis = 1)**2 + 1.E-16j
    return np.sum(numerator/denominator).real# .real
    # #Check for zero denominator.
    # if sum(denominator) == 0.:
    #     return np.nan
    # else: 
    #     return sum(numerator/denominator)

def disparitylattice(matrix):
    #Local disparity across the entire lattice.
    numerator = np.sum(matrix**2, axis = 1)
    denominator = np.sum(matrix, axis = 1)**2 + 1.E-16j
    disparity = (numerator/denominator).real
    return disparity

def strengthdist(mutualinformation, bins = np.linspace(0, 1, 11)):
    #Compute the weighted analog of a degree distribution.
    strens = strengths(mutualinformation)
    # maxinfo=np.max(strens)
    # mininfo=np.min(strens)
    return np.histogram(strens,bins)

def pearson(mutualinformation):
    L = len(mutualinformation)
    pearsonR = np.zeros((L,L))
    pvalues = np.zeros((L,L))
    for i in range(L):
        for j in range(L):
            if i != j:
                r,p = stats.pearsonr(mutualinformation[i], mutualinformation[j])
                pearsonR[i][j] = r
                pvalues[i][j] = p
            else:
                pearsonR[i][j] = 0.
                pvalues[i][j] = 0.
    return [pearsonR, pvalues]

def edgeweightdist(mutualinformation, bins):
    L = len(mutualinformation)
    edges = es.flatten(list(mutualinformation))
    return np.histogram(edges,bins)[0]

def coarsegrainnetwork(mutualinformation, bins):
    #Default behavior of histograms is to have the intervals
    #Closed on the left and open on the right.
    #The bin with the largest values is closed on both ends.
    L = len(mutualinformation)
    coarsematrix = np.zeros((L,L))
    ll = len(bins)
    for i in range(L):
        for j in range(L):
            for k in range(ll-2):
                if bins[k] <= mutualinformation[i][j] < bins[k+1]:
                    coarsematrix[i][j] = k
            if bins[ll-2] <= mutualinformation[i][j] <= bins[ll-1]:
                coarsematrix[i][j] = ll-2
    return coarsematrix

#NetworkX additions
#---------------------------------------------

def distance(mutualinformation):
    #Initialize array
    length = len(mutualinformation)
    thisdistance = np.zeros((length,length))
    #If an element has value less than (10^-14) in absolute value,
    #then treat it as of value (10^-16), set its distance
    #to 10^16. Otherwise set it mij^(-1).
    #The value (10^-14) must be adjusted in coordination with
    #the cutoff applied to the weighted network of interest.
    #The purpose of setting this value equal to 10^16 is
    #so that the geodesic path length algorithm will ignore
    #any edges less than the cutoff treating.
    for i in range(length):
        for j in range(length):
            if np.abs(mutualinformation[i, j]) <= 1E-14:
                thisdistance[i, j] = np.power(10.,16)
            else:
                thisdistance[i, j] = np.power(mutualinformation[i, j],-1)
    return thisdistance

def geodesic(distance, i, j):
    #Initialize networkx graph object
    #NetworkX indexing starts at zero.
    #Ends at N-1 where N is the number of nodes.
    latticelength = len(distance)
    G = nx.Graph(distance)
    #Use networkx algorithm to compute the shortest path from
    #the first lattice site to the last lattice site.
    pathlength = nx.shortest_path_length(G, source = i-1, target = j-1, weight = 'weight')
    #The pathlength is tested for an unreasonable value.
    #An unreasonable value for the path length is a function of the cutoff
    #applied to the weighted network under analysis and the value set in
    #the distance function thisdistance[i,j]=np.power(10.,16)
    if pathlength > np.power(10., 15):
        return np.nan
    return pathlength

def harmoniclength(distance):
    #page 11, equation 2 The Structure and Function of Complex Networks
    #If the geodesic distance between two nodes is a number then 
    #append it to alist to include it in the sum.
    l = len(distance)
#    print 'Node count: ', l
    factor = 1./(0.5*l*(l-1))
#    print 'Factor: ', factor
    alist = []
    for i in range(1, len(distance) + 1):
        for j in range(i+1, len(distance) + 1):
            geo = geodesic(distance, i, j)
            if not np.isnan(geo):
#                print '(i,j)',tuple([i,j])
                alist.append(1./geo)
#    print 'alist',alist
#    print 'Harmonic Mean of Distances: ', 1./(factor*sum(alist))
    return 1./(factor*sum(alist))

def eigenvectorcentralitynx0(mutualinformation):
    #Uses the power method to find eigenvector of 
    #a weighted adjacency matrix.
    G = nx.Graph(mutualinformation)
    eigvcent = nx.eigenvector_centrality(G, weight='weight', max_iter=2000)
    return eigvcent

def eigenvectorcentralitynx(mutualinformation, startingvector):
    #Identical to eigenvectorcentralitynx0, but requires an additional argument startingvector.
    #starting vector provides an initial guess for the eigen vector centrality of all nodes.
    #startingvector must be a python dictionary. key = node, value = eigenvector centrality estimate.
    G = nx.Graph(mutualinformation)
    eigvcent = nx.eigenvector_centrality(G, weight = 'weight', max_iter = 2000, nstart = startingvector)
    return eigvcent

#Here I provide some test cases.  These test cases assume the weighted
#networks being provided are quantum mutual information networks.
test=True
def main():
    if test==True:
        testrho = np.array([[0.1, 0], [0, 0.9]])
        testrho2 = np.array([[0.7, 0], [0, 0.3]])
        testrho3 = np.array([[0.5, 0.], [0.0, 0.5]])
        testrhos = [testrho, testrho2, testrho3]
        singlet = np.array([[0., 0., 1.], [0., 0., 0.], [1., 0., 0.]])
        ghz = np.array([[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
        w = np.array([[0, 0.636514, 0.636514],
                    [0.636514, 0, 0.636514],
                    [0.636514, 0.636514, 0]])
        fourqubits = np.array([[0, 1, 0, 0], [1, 0, 1, 1],
                             [0, 1, 0, 1], [0, 1, 1, 0]])
        states = [singlet, ghz, w, fourqubits]
        names = ['singlet', 'ghz', 'w', 'Four Qubits']
        functions = [strengths, density, clustering, localclustering, disparity,
                     disparitylattice, distance]#, eigenvectorcentralitynx0]
        measurenames = ['Strengths', 'Density', 'Clustering', 
                        'Local Clustering', 'Average Disparity', 
                        'Nodal Disparity', 'Distances']#,
#                        'Eigenvector Centrality']
        for i in range(len(states)):
            print(names[i])
            for function in functions:
                print(function, function(states[i]))
        print('Rhos: ', testrho, '\n', testrho2, '\n', testrho3, '\n')
        print('Rho Types: ', rhotypes(testrhos))
        print(quantumassortativity(testrhos, ghz, rhotypes))
        #Counterintuitive result for pearson correlation of ghz state.
        #It is negative. This is because we are arbitarily setting the 
        #diagonal of the mutual information matrix to zero.  
        print(pearson(ghz))
        
        print("Edge Weight Distribution: ", edgeweightdist(ghz, [0, 0.5, 1]))
        print("Full Matrix: ", ghz)
        print("Coarse Matrix:", coarsegrainnetwork(ghz, [0, 0.5, 1]))
        
        
        # print("Local Clustering: ", localclustering(matrix))
        
        # total = 0
        # total2 = 0
        # for i in range(10000):
        #     matrix = np.zeros((26, 26))
        #     for i in range(26):
        #         for j in range(26):
        #             matrix[i][j] = random.random()
        #     t0 = time.time()
        #     clustering(matrix)
        #     t1 = time.time()
        #     clustering2(matrix)
        #     t2 = time.time()
        #     total += (t1-t0)/(t2-t1)
        #     total2 += clustering2(matrix) - clustering(matrix)
        # print(total/10000)
        # print(total2/10000)
        #This effect should go away in the thermodynamic limit. 
        #I can check this. Next I should check how robust the
        #result is against the size of the numerical noise.
        # def GHZ(L):
        #     mutualinformation=np.zeros((L,L))
        #     for i in range(L):
        #         for j in range(L):
        #             if i!=j:
        #                 mutualinformation[i][j]=0.5#+0.01*(np.random.rand()-0.5)
        #             else:
        #                 mutualinformation[i][j]=0.
        #     return mutualinformation
        # for l in [4,10,20,50]:
        #     print l,'\n',pearson(GHZ(l))[0][l/2][l/2+1]
        # a=1.E-14*np.ones((20,20))
        # for i in range(len(a)):
        #     a[i][i]=0.
        # print 'disparity', disparity(a)
        # print 'clustering', clustering(GHZ(3))
        # print 0.2*GHZ(3)
        # a=0.01*np.ones(100)*range(100)
        # for i in a:
        #     print clustering(i*GHZ(3))

if __name__ == "__main__":
    main()
