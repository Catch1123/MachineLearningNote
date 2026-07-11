#%%
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv


def on_site_cross_entropy(probs, labels):
    """
    On-site cross entropy loss.
    
    Parameters:
        probs : np.ndarray, shape (batch_size, num_classes)
            Softmax output probabilities.
        labels : np.ndarray, shape (batch_size,)
            True labels as integers {0, 1, ..., 9}.
    
    Returns:
        loss : float
            Scalar cross entropy loss averaged over batch.
    
    Mathematically:
        Loss = -1/N * Σ_i log( Σ_c δ(y_i - c) * p_{i,c} )
    """
    batch_size = probs.shape[0]
    
    # On-site projection: select probability at the true label column
    on_site_probs = probs[np.arange(batch_size), labels]
    
    # Cross entropy: -log(on_site_prob)
    loss = -np.mean(np.log(on_site_probs + 1e-12))
    
    return loss

def fd(beta,L,R):
    return (np.exp(-beta* (L @ R) ) + 1)**(-1)


train_data = tv.datasets.MNIST(root='./data',train=True,download=True)
test_data = tv.datasets.MNIST(root='./data',train=False,download=True)

#data per-process and adding bias
td = train_data.data.numpy()
la = train_data.targets.numpy()
td = td.reshape(-1,28*28)/225
bias = np.ones((td.shape[0],1))
td   = np.hstack([td,bias])
#%%
# ============ Neural Network Initialization ============

# Network architecture
layer_dims = [td.shape[1], 50, 30, 10]   # [785, 50, 30, 10]

# Inverse temperature for each layer (Fermi-Dirac distribution)
beta = [1.0, 1.0, 1.0]                   # [beta1, beta2, beta3]

# Xavier initialization
def init_weights(dim_in, dim_out):
    return np.random.randn(dim_in, dim_out) * np.sqrt(1.0 / dim_in)

# Initialize weights
W1 = init_weights(layer_dims[0], layer_dims[1])   # (785, 50)
W2 = init_weights(layer_dims[1], layer_dims[2])   # (50, 30)
W3 = init_weights(layer_dims[2], layer_dims[3])   # (30, 10)

weights = [W1, W2, W3]
#%%
# ============ Forward pass ============
# Layer 1: input -> hidden layer 1
Af1 = fd(beta[0], td, W1)      # td @ W1 -> Sigmoid

# Layer 2: hidden layer 1 -> hidden layer 2
Af2 = fd(beta[1], Af1, W2)     # Af1 @ W2 -> Sigmoid

# Layer 3: hidden layer 2 -> output layer
Af3 = fd(beta[2], Af2, W3)     # Af2 @ W3 -> Sigmoid

# Softmax: convert scores to probabilities
exAf3 = np.exp(Af3) / np.sum(np.exp(Af3), axis=1, keepdims=True)

# On-site cross entropy loss
loss = on_site_cross_entropy(exAf3, la)
# %%
