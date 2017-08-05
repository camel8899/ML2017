import numpy as np 
import sys
import os
from numpy.random import normal as N 
from sklearn.metrics.pairwise import euclidean_distances as ed
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVR as SVR

def elu(arr):
    return np.where(arr > 0, arr, np.exp(arr) - 1)

def make_layer(in_size, out_size):
    w = np.random.normal(scale=0.5, size=(in_size, out_size))
    b = np.random.normal(scale=0.5, size=out_size)
    return (w, b)

def forward(inpd, layers):
    out = inpd
    for layer in layers:
        w, b = layer
        out = elu(out @ w + b)
    return out

def gen_data(dim, layer_dims, N):
    layers = []
    data = np.random.normal(size=(N, dim))

    nd = dim
    for d in layer_dims:
        layers.append(make_layer(nd, d))
        nd = d

    w, b = make_layer(nd, nd)
    gen_data = forward(data, layers)
    gen_data = gen_data @ w + b
    return gen_data


def get_eigenvalues(data):
    SAMPLE = 100 # sample some points to estimate
    NEIGHBOR = 200 # pick some neighbor to compute the eigenvalues
    randidx = np.random.permutation(data.shape[0])[:SAMPLE]
    knbrs = NearestNeighbors(n_neighbors=NEIGHBOR,
                             algorithm='ball_tree').fit(data)

    sing_vals = []
    for idx in randidx:
        dist, ind = knbrs.kneighbors(data[idx:idx+1])
        nbrs = data[ind[0,1:]]
        u, s, v = np.linalg.svd(nbrs - nbrs.mean(axis=0))
        s /= s.max()
        sing_vals.append(s)
    sing_vals = np.array(sing_vals).mean(axis=0)
    return sing_vals


#Method 2 in report, TA's method
# generate some data for training
'''
X = []
y = []
for i in range(60):
    dim = i + 1
    #print("=======================Dimension "+str(dim)+"=======================")
    for N in [10000, 20000,50000,80000,100000]:
        #print("Generating data with size " +str(N))
        layer_dims = [np.random.randint(60, 80), 100]
        data = gen_data(dim, layer_dims, N).astype('float32')
        eigenvalues = get_eigenvalues(data)
        X.append(eigenvalues)
        y.append(dim)

X = np.array(X)
y = np.array(y)

np.savez('train_data', X=X, y=y)
'''

# Train a linear SVR

loadfile = np.load('train_data.npz')
X = loadfile['X']
y = loadfile['y']
svr = SVR(C=4.5)
svr.fit(X, y)

# predict

testdata = np.load(sys.argv[1])
test_X = []
for i in range(200):
    data = testdata[str(i)]
    vs = get_eigenvalues(data)
    test_X.append(vs)

test_X = np.array(test_X)
pred_y = svr.predict(test_X)
pred_y = [round(i) for i in pred_y.tolist()]

#Output

with open(sys.argv[2], 'w') as f:
    print('SetId,LogDim', file=f)
    for i, d in enumerate(pred_y):
        print('{},{}'.format(i,np.log(d)), file=f)



#My first method in report, no need to run this
'''

def nearest_distance(size,data):
    d_matrix = ed(data[0:10000],data[0:10000])
    np.fill_diagonal(d_matrix,float('inf'))
    #k nearest
    k = 1
    return np.array([np.mean([d_matrix[i][np.argpartition(d_matrix[i],k)[:k]]]) for i in range(size)])

def Std_of_distance(dim):
    N = 10000
    layer_dims = [np.random.randint(60, 80), 100]
    print("Generating Data of dimension " + str(dim))
    data = gen_data(dim, layer_dims, N)
    print("Generating k_nearset distance list of dimension " + str(dim))
    distance_list = nearest_distance(5000,data)
    return np.std(distance_list)

Std = []
for i in range(1,61):
    print("Calculating the std of distance of dimension " + str(i))
    Std.append(Std_of_distance(i))

with open('Std','w') as f:
    f.write('SetId,Std_of_distance\n')
    for idx,a in enumerate(Std):
        f.write('{},{}\n'.format(idx,a))

data = np.load(sys.argv[1])
Std_true = []
with open('Std','r') as f:
    for line in f.readlines():
        if line.split(',')[0] == 'SetId':
            continue
        else:
            Std_true.append(float((line.strip().split(',')[1])))

ans = []
Std_calculate = []
for i in range(200):
    x = data[str(i)]
    distance_list = nearest_distance(5000,x)
    print("Successfully generate distance list for dataset "+str(i))
    Std_calculate = np.std(distance_list)
    print("Successfully calculate std of distance for dataset "+str(i))
    min_d = float("inf")
    ans_id = 0
    for idx,s in enumerate(Std_true):
        if np.abs(Std_calculate-s) < min_d:
            print("Change!")
            min_d = np.abs(Std_calculate-s)
            ans_id = idx
    ans.append(ans_id)

with open(sys.argv[2],'w') as f:
    f.write('SetId,LogDim\n')
    for idx,a in enumerate(ans):
        f.write('{},{}\n'.format(idx,np.log(float(a)+1)))
'''

