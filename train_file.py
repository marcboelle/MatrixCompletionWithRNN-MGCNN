import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import h5py
import scipy.sparse as sp

from code.models import LaplacianConvolution, RecurrentsGNN
from code.utils import compute_cheb_polynomials
from code.train import train

"""n_user = 10 #nodes
n_item=13
m=20 #edges
seed=0

G_user = nx.gnm_random_graph(n_user, m, seed=seed)
G_item = nx.gnm_random_graph(n_item, m, seed=seed)

laplacian_user = nx.laplacian_matrix(G_user).toarray()
laplacian_item = nx.laplacian_matrix(G_item).toarray()

r=7
W = np.ones((n_user, r), dtype = np.float32)
H = np.ones((n_item, r), dtype = np.float32) #when transposed give the (r, n_item) matrix 

#list_lap = compute_cheb_polynomials(laplacian, ord_cheb = 3)

#model = LaplacianConvolution(list_lap, out_dim = 5, rank_matrix = 7)
model = RecurrentsGNN(laplacian_user, laplacian_item, order_chebyshev_col=2, order_chebyshev_row=2, n_conv_feat=32,
                      lstm_hidden_size=30, rank_matrix=r)"""

def load_matlab_file(path_file, name_field):
    """
    load '.mat' files
    inputs:
        path_file, string containing the file path
        name_field, string containig the field name (default='shape')
    warning:
        '.mat' files should be saved in the '-v7.3' format
    """
    db = h5py.File(path_file, 'r')
    ds = db[name_field]
    try:
        if 'ir' in ds.keys():
            data = np.asarray(ds['data'])
            ir   = np.asarray(ds['ir'])
            jc   = np.asarray(ds['jc'])
            out  = sp.csc_matrix((data, ir, jc)).astype(np.float32)
    except AttributeError:
        # Transpose in case is a dense matrix because of the row- vs column- major ordering between python and matlab
        out = np.asarray(ds).astype(np.float32).T

    db.close()

    return out


path_dataset = 'mgcnn/Data/movielens/split_1.mat'

#loading of the required matrices
M = load_matlab_file(path_dataset, 'M')
Otraining_init = load_matlab_file(path_dataset, 'Otraining')
Otest = load_matlab_file(path_dataset, 'Otest')
Wrow = load_matlab_file(path_dataset, 'W_users') #sparse
Wcol = load_matlab_file(path_dataset, 'W_movies') #sparse


np.random.seed(0)
pos_tr_samples = np.where(Otraining_init) #returns indices of non-zero values in Otraining, (row_indexes, col_indexes)
num_tr_samples = len(pos_tr_samples[0]) #number of non-zero values 
list_idx = range(num_tr_samples)
list_idx = np.random.permutation(list_idx)
idx_data = list_idx[:num_tr_samples//2] #half of the samples, drawn randomly
idx_train = list_idx[num_tr_samples//2:] #other half

pos_data_samples = (pos_tr_samples[0][idx_data], pos_tr_samples[1][idx_data]) #selection of a random half of the non-zero coefs of Otraining
pos_tr_samples = (pos_tr_samples[0][idx_train], pos_tr_samples[1][idx_train]) #other half

Odata = np.zeros(M.shape)
Otraining = np.zeros(M.shape)

for k in range(len(pos_data_samples[0])):
    Odata[pos_data_samples[0][k], pos_data_samples[1][k]] = 1
    
for k in range(len(pos_tr_samples[0])):
    Otraining[pos_tr_samples[0][k], pos_tr_samples[1][k]] = 1


Lrow = sp.csgraph.laplacian(Wrow, normed=True)
values = Lrow.data
indices = np.vstack((Lrow.row, Lrow.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = Lrow.shape
Lrow = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense() #for test

Lcol = sp.csgraph.laplacian(Wcol, normed=True)
values = Lcol.data
indices = np.vstack((Lcol.row, Lcol.col))
i = torch.LongTensor(indices)
v = torch.FloatTensor(values)
shape = Lcol.shape
Lcol = torch.sparse_coo_tensor(i, v, torch.Size(shape)).to_dense() #for test

U, s, V = np.linalg.svd(Odata*M, full_matrices=0)
#Odata*M is a half of the non-zero 

rank_W_H = 10
partial_s = s[:rank_W_H]
partial_S_sqrt = np.diag(np.sqrt(partial_s))
initial_W = np.dot(U[:, :rank_W_H], partial_S_sqrt) #items 
initial_H = np.dot(partial_S_sqrt, V[:rank_W_H, :]).T

model = RecurrentsGNN(Lrow, Lcol, order_chebyshev_col=2, order_chebyshev_row=2, n_conv_feat=32,
                      lstm_hidden_size=30, rank_matrix=10)


train(model, M, Lrow, Lcol,Odata=Odata, Otraining=Otraining, Otest=Otest, initial_W=initial_W,
      initial_H=initial_H, gamma=1, learning_rate=0.0001, epochs=100, verbose=10)


