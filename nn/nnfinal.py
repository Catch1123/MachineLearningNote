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

def fd_derivative(beta, A_prev, W):
    """Derivative of Fermi-Dirac activation: β * f * (1 - f)"""
    z = A_prev @ W
    f = 1 / (1 + np.exp(-beta * z))
    return beta * f * (1 - f)

train_data = tv.datasets.MNIST(root='./data',train=True,download=True)
test_data = tv.datasets.MNIST(root='./data',train=False,download=True)

#data per-process and adding bias
td = train_data.data.numpy()
la = train_data.targets.numpy()
td = td.reshape(-1,28*28)/225
bias = np.ones((td.shape[0],1))
td   = np.hstack([td,bias])

# ============ Neural Network Initialization ============

# Network architecture
layer_dims = [td.shape[1], 80, 10]   # [785, 100, 10]

# Inverse temperature for each layer (Fermi-Dirac distribution)
beta = np.array([1.0, 1.0, 1.0]) * 3.2                # [beta1, beta2, beta3]

# Xavier initialization
def init_weights(dim_in, dim_out):
    return np.random.randn(dim_in, dim_out) * np.sqrt(1.0 / dim_in)

# Initialize weights
W1 = init_weights(layer_dims[0], layer_dims[1])   # (785, 100)
# W2 = init_weights(layer_dims[1], layer_dims[2])   # (100, 10)
W3 = init_weights(layer_dims[1], layer_dims[2])   # (100, 10)

weights = [W1,W3]

# ============ Training Hyperparameters ============
epochs = 300          # the number of training epochs
learning_rate = 5e-2  # learning rate for gradient descent
batch_size = 64      # mini-batch size

loss_history = []    # record loss for each epoch

for epoch in range(epochs):
    epoch_loss = 0
    num_batches = 0
    
    # ---------- Mini-batch training ----------
    for i in range(0, len(td), batch_size):
        # Get current batch
        batch_x = td[i:i+batch_size]
        batch_y = la[i:i+batch_size]
        batch_size_current = batch_x.shape[0]
        
        #============ Forward pass ============
        # Layer 1: input -> hidden layer 1
        Af1 = fd(beta[0], batch_x, W1)      # batch_x @ W1 -> Fermi-Dirac
        
        # # Layer 2: hidden layer 1 -> hidden layer 2
        # Af2 = fd(beta[1], Af1, W2)          # Af1 @ W2 -> Fermi-Dirac
        
        # Layer 3: hidden layer 2 -> output layer
        # Af3 = fd(beta[2], Af1, W3)          # Af2 @ W3 -> Fermi-Dirac
        Af3 = Af1 @ W3         # Af2 @ W3 -> Fermi-Dirac
        
        # Softmax: convert scores to probabilities
        exAf3 = np.exp(Af3)
        probs = exAf3 / np.sum(exAf3, axis=1, keepdims=True)
        
        # On-site cross entropy loss
        loss = on_site_cross_entropy(probs, batch_y)
        epoch_loss += loss
        num_batches += 1
        
        #============= Backward pass ==============
        # Gradient of loss w.r.t. pre-activation output (softmax + cross entropy)
        one_hot = np.eye(10)[batch_y]  # (batch_size, 10)
        dL_dAf3 = probs - one_hot      # (batch_size, 10)
        
        # Layer 3 (output layer)
        dL_dz3 = dL_dAf3 * fd_derivative(beta[2], Af1, W3)  # (batch_size, 10)
        dL_dW3 = Af1.T @ dL_dz3 / batch_size_current        # (100, 10)
        dL_dAf1 = dL_dz3 @ W3.T                             # (batch_size, 100)
        
        # # Layer 2 (hidden layer 2)
        # dL_dz2 = dL_dAf1 * fd_derivative(beta[1], Af1, W1)  # (batch_size, 50)
        # dL_dW2 = Af1.T @ dL_dz2 / batch_size_current        # (50, 30)
        # dL_dAf1 = dL_dz2 @ W1.T                             # (batch_size, 785)
        
        # Layer 1 (hidden layer 1)
        dL_dz1 = dL_dAf1 * fd_derivative(beta[0], batch_x, W1)  # (batch_size, 785)
        dL_dW1 = batch_x.T @ dL_dz1 / batch_size_current        # (785, 100)
        
        #============= Update weights ==============
        W1 -= learning_rate * dL_dW1
        # W2 -= learning_rate * dL_dW2
        W3 -= learning_rate * dL_dW3
    
    # ---------- End of epoch ----------
    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss)
    
    # Print progress
    print(f"Epoch {epoch+1:3d}/{epochs} | Avg Loss: {avg_loss:.6f}")



# ============ Evaluation on Test Set ============

# Prepare test data
td_test = test_data.data.numpy()
la_test = test_data.targets.numpy()
td_test = td_test.reshape(-1, 28*28) / 225
bias_test = np.ones((td_test.shape[0], 1))
td_test = np.hstack([td_test, bias_test])

# Forward pass on test data
Af1_test = fd(beta[0], td_test, W1)
# Af2_test = fd(beta[1], Af1_test, W2)
Af3_test = fd(beta[2], Af1_test, W3)
exAf3_test = np.exp(Af3_test)
probs_test = exAf3_test / np.sum(exAf3_test, axis=1, keepdims=True)

# Predictions and accuracy
predictions = np.argmax(probs_test, axis=1)
accuracy = np.mean(predictions == la_test)

print(f"\n测试集准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============ Visualize Training Loss ============

plt.figure(figsize=(10, 6))
plt.plot(loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss vs. Epoch', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()

# ============ Optional: Visualize some predictions ============

# Show first 20 test images with predictions
fig, axes = plt.subplots(8, 5, figsize=(12, 6))
axes = axes.ravel()

for idx in range(40):
    # Get image and prediction
    img = test_data.data[idx].numpy()
    pred = predictions[idx]
    true_label = la_test[idx]
    
    axes[idx].imshow(img, cmap='gray')
    axes[idx].set_title(f'Pred: {pred}, True: {true_label}', 
                        color='green' if pred == true_label else 'red')
    axes[idx].axis('off')

plt.suptitle('Test Set Predictions', fontsize=14)
plt.tight_layout()
plt.show()