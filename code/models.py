import torch.nn as nn
import torch
import numpy as np

from .utils import compute_cheb_polynomials

class LaplacianConvolution(nn.Module):
    """
    Returns a convolution between spectral filters and a matrix
    """

    def __init__(self, tensor_laplacians : torch.Tensor, out_dim : int, rank_matrix : int):
        super(LaplacianConvolution, self).__init__()

        #scalar values
        self.rank_matrix = rank_matrix #rank of the matrix (for sGCNN with H and W it is r)
        self.out_dim = out_dim

        #precomputed list of the laplacians used as spectral filters for convolution
        self.tensor_laplacians = tensor_laplacians

        #operations
        self.nb_filters, self.n_nodes, _  = tensor_laplacians.shape
        self.linear = nn.Linear(self.nb_filters * self.rank_matrix, self.out_dim)
        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        #Xavier uniform initialization        
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, A):
        indiv_conv = self.tensor_laplacians @ A #shape (cheb_order, A.shape[0], out_dim)

        indiv_conv = indiv_conv.permute(1, 0, 2).reshape(self.n_nodes, self.nb_filters * self.rank_matrix) #shape (A.shape[0], cheb_order * rank_matrix), ready for linear layer

        out_conv = self.linear(indiv_conv)
        out_conv = self.relu(out_conv)

        return out_conv





#the model used for matrix completion

class RecurrentsGNN(nn.Module):
    """
    Follows the work of Monti et al., "Geometric Matrix completion with Recurrent Multi-Graph Neural Networks
    """

    def __init__(self, Lr : np.array, Lc : np.array,  rank_matrix : int,
                 order_chebyshev_col : int=5, order_chebyshev_row : int=5, n_conv_feat : int=32, lstm_hidden_size : int=32,
                 num_iterations : int=10):
        super(RecurrentsGNN, self).__init__()

        #Initialize scalar values
        self.ord_col = order_chebyshev_col
        self.ord_row = order_chebyshev_row
        self.num_iterations = num_iterations
        self.n_conv_feat = n_conv_feat
        self.lstm_hidden_size = lstm_hidden_size
        self.rank_matrix = rank_matrix
        
        #Load the Laplacians as tensors
        if isinstance(Lr, np.ndarray):
            Lr = torch.Tensor(Lr)
        if isinstance(Lc, np.ndarray):
            Lc = torch.Tensor(Lc)

        self.Lr = Lr
        self.Lc = Lc

        #Laplacians normalized to have eigenvalues between -1 and 1

        self.norm_Lr = self.Lr - torch.eye(Lr.shape[0])
        self.norm_Lc = self.Lc - torch.eye(Lc.shape[0])

        #Compute Chebyshev polynomials a priori
        self.list_row_cheb_pol = compute_cheb_polynomials(self.norm_Lr, self.ord_row)
        self.list_col_cheb_pol = compute_cheb_polynomials(self.norm_Lc, self.ord_col)

        #Initialize layers of the NN
        self.row_convol = LaplacianConvolution(self.list_row_cheb_pol, self.n_conv_feat, self.rank_matrix)
        self.col_convol = LaplacianConvolution(self.list_col_cheb_pol, self.n_conv_feat, self.rank_matrix)
        self.row_lstm = nn.LSTM(self.n_conv_feat, self.lstm_hidden_size)        #automatically initialized uniformly in (-sqrt(k), sqrt(k)) with k = 1/lstm_hidden_size
        self.col_lstm = nn.LSTM(self.n_conv_feat, self.lstm_hidden_size)    
        self.final_linear_row = nn.Linear(self.lstm_hidden_size, self.rank_matrix)  #final linear layer to obtain a matrix of the same shape as the initial one for update
        self.final_linear_col = nn.Linear(self.lstm_hidden_size, self.rank_matrix)  #final linear layer to obtain a matrix of the same shape as the initial one for update
        self.tanh = nn.Tanh()

    def forward(self, initial_W : torch.Tensor, initial_H : torch.Tensor):
        if isinstance(initial_W, np.ndarray):
            initial_W = torch.Tensor(initial_W)
        if isinstance(initial_H, np.ndarray):
            initial_H = torch.Tensor(initial_H)

        W, H = initial_W, initial_H
        list_X = []
        for k in range(self.num_iterations):
            user_features = self.row_convol(W)
            item_features = self.col_convol(H)

            _, (_, user_c_sequence) = self.row_lstm(user_features) #user_c_sequence of shape (1, lstm_hidden_size)
            _, (_, item_c_sequence) = self.col_lstm(item_features) #shape (1, lstm_hidden_size)
            user_output = self.tanh(self.final_linear_row(user_c_sequence[-1])) 
            item_output = self.tanh(self.final_linear_col(item_c_sequence[-1]))

            #update item and user matrices
            W += user_output
            H += item_output

            X = W @ H.T
            list_X.append(X)
        return X, list_X

        