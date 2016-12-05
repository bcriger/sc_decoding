import cPickle as cp
import numpy as np

with open('data_rn.pkl', 'r') as f:
    data = cp.load(f)

l=8
N=20000

logicals=np.asarray(data['logicals'][:N])

rn_current1=np.asarray(data['rn_current1'][:N], dtype=bool)

# anyons_boundary=np.zeros((N,l,l,3),dtype=bool)
anyons_boundary=np.zeros((N,l,l,2),dtype=bool)
# all data not near the bnd should have bnd value 0
anyons_boundary[:,:,:,0]=np.stack(data['anyons'][:N],axis=0)

# mark the two different boundary
# the data in the bnd should have bnd value 1, two bnds one at 0 and one at l-1
anyons_boundary[:,:,0,1]=1
anyons_boundary[:,:,l-1,1]=1

# anyons_boundary[:,0,:,2]=1
# anyons_boundary[:,l-1,:,2]=1

with open('processed_data_rn_8_0.05_20000.pkl','w') as f:
    cp.dump({'anyons':anyons_boundary, 'rn_current1':rn_current1, 'logicals':logicals},f)