import torch
import torch.nn as nn
import numpy as np
from typing import Union
from tqdm import tqdm

from code.utils import frobenius_norm


def train(model : nn.Module , M : np.array, Lr : np.array, Lc : np.array, Odata : np.array, Otraining : np.array, Otest : np.array, initial_W : np.array, initial_H : np.array,
         gamma=1.0, learning_rate=1e-4, epochs : int=5000, verbose : Union[None, int]=None):

    if isinstance(M, np.ndarray):
        M = torch.Tensor(M)
    if isinstance(Odata, np.ndarray):
        Odata = torch.Tensor(Odata)
    if isinstance(Otraining, np.ndarray):
        Otraining = torch.Tensor(Otraining)
    if isinstance(Otest, np.ndarray):
        Otest = torch.Tensor(Otest)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = [] #containing frobenius + regularization losses
    frobenius_train_losses = []
    frobenius_test_losses = []
    for epoch in tqdm(range(epochs)):
        X, list_X = model(initial_W, initial_H)

        # Normalization of X (for MovieLens, because coeffs are between 1 and 5)
        norm_X = 1 + 4 * (X - torch.min(X)) / (torch.max(X) - torch.min(X))

        # Compute the accuracy term
        frob_tensor = (Otraining + Odata) * (norm_X - M)
        loss_frob = frobenius_norm(frob_tensor) ** 2 / torch.sum(Otraining + Odata)

        # Compute the regularization terms
        trace_col_tensor = torch.matmul(torch.matmul(X, Lc), X.T)
        loss_trace_col = torch.trace(trace_col_tensor)
        
        trace_row_tensor = torch.matmul(torch.matmul(X.T, Lr), X)
        loss_trace_row = torch.trace(trace_row_tensor)

        # Define the total training loss
        loss = loss_frob + (gamma / 2) * (loss_trace_col + loss_trace_row)
        train_losses.append(loss.item())
        frobenius_train_losses.append(loss_frob.item())
        
        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the test error
        with torch.no_grad():
            predictions = Otest * (norm_X - M)
            predictions_error = frobenius_norm(predictions)

        frobenius_test_losses.append(predictions_error)

        
        if verbose != None and (epoch+1)%verbose==0:
            print(f"Epoch {epoch} - Total train loss : {round(loss.item(), 2)} - Frob train loss : {round(loss_frob.item(), 2)} - Frob test loss : {round(predictions_error.item(), 2)}")
    
    return train_losses, frobenius_train_losses, frobenius_test_losses
        