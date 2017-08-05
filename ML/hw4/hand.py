import numpy as np 
import sys
import os
from numpy.random import normal as N 
from sklearn.neighbors import NearestNeighbors
from PIL import Image
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

def nearest_distance(size,data):
    d_matrix = ed(data[0:10000],data[0:10000])
    np.fill_diagonal(d_matrix,float('inf'))
    k = 5   
    return np.array([np.mean([d_matrix[i][np.argpartition(d_matrix[i],k)[:k]]]) for i in range(size)])

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

npzfile = np.load('train_data.npz')
X = npzfile['X']
y = npzfile['y']

# we already normalize these values in gen.py
# X /= X.max(axis=0, keepdims=True)

svr = SVR(C=4.5)
svr.fit(X, y)

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
hand_path = os.path.join(base_path,'hand')

img = []
for file in os.listdir(hand_path):
    if file.split('.')[0] == 'hand':
        img.append(np.reshape(np.array(Image.open(os.path.join(hand_path,file)).resize((10,10),Image.BILINEAR)),(100)))
        
img = np.array(img)
img_eigenmean = get_eigenvalues(img)

pred_y = svr.predict(img_eigenmean)
print("The dimension of this hand dataset is " + str(round(int(pred_y))))

