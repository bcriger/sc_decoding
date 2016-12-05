import random
import cPickle as cp
import numpy as np

L = 8
p = 0.05

anyons_set = []
logicals_set = []
rn_current1_set=[]

for num in range(20000):

    # before noise, there are no errors and no anyons
    # anyons = [[0 for col in range(10)] for row in range(10)]
    anyons = np.zeros((L, L))

    # and no logical errors
    logicals = 0

    rn_current1= np.zeros((L/2,L/2,2))

    # loop over (almost) all plaquettes
    for y in range(L):
        for x in range(L):

            """
            there are twice as many qubits as plaquettes, so for each
            plaquette we choose two qubits. These will be the one to
            the right, and the one to the bottom
            """

            # first we randomly decide whether to place a bit flip error to the right
            if (random.random() < p):
                # syndrome is flipped at (x,y) and (x+1,y)
				# generate a pair of anyons when the prob < p, which happens by flipping the value of [x][y]
                anyons[x][y] = (anyons[x][y] + 1) % 2

                # if y+1 is out of range, the anyon goes off the right edge and is unrecorded
				# if the anyon that was flipped was not in the boundary, then flip its neighbour too, to create a pair
                if ((y + 1) < L):
                    anyons[x][y+1] = (anyons[x][y+1] + 1) % 2
					# check if the boundary is the correct one, in this case the one one the left
                    if y%2==1:
                        rn_current1[x//2,y//2+1,0]+=1  # the third co-ordinate 0 corresponds to the left edge

        # no possibility to go off edge in this direction, so loop ensures y+1<L
        for x in range(L - 1):

            # now the same for a bit flip to the bottom
            if (random.random() < p):
                #  syndrome is flipped at (x,y) and (x,y+1)
                anyons[x][y] = (anyons[x][y] + 1) % 2
                anyons[x+1][y] = (anyons[x+1][y] + 1) % 2

                if x%2==1:
                    rn_current1[x//2,y//2,1]+=1  # the third co-ordinate 1 corresponds to the down edge

    # not all errors have yet been done. First loop over the right edge
    for x in range(L):
        # add errors across edge in the same way as above
        if (random.random() < p):
            anyons[x][0] = (anyons[x][0] + 1) % 2
            # errors across edge in this case are logical errors, and are recorded as such

            rn_current1[x//2,0,0]+=1
            logicals = (logicals + 1) % 2

    # This is just something to display the array that I got from here
    # http://stackoverflow.com/questions/13214809/pretty-print-2d-python-list
    # s = [[str(e) for e in row] for row in anyons]
    # lens = [max(map(len, col)) for col in zip(*s)]
    # fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    # table = [fmt.format(*row) for row in s]
    # print '\n'.join(table)
    #
    #
    # print anyons
    # print rn_current1[:,:,0] % 2
    # print rn_current1[:,:,1] % 2
    # print logicals

    anyons_set.append(anyons)
    rn_current1_set.append(rn_current1 % 2)
    logicals_set.append(logicals)

with open('data_rn.pkl','w') as f:
    cp.dump({'anyons':anyons_set, 'rn_current1':rn_current1_set, 'logicals':logicals_set},f)