import numpy as np
from network import relu_derivative

def backward_pass(params, cache, labels):
    """
    Backpropagation — computes gradients of the loss with respect
    to every weight and bias in the network.
    
    We work backwards from the output layer to the input layer
    using the chain rule from calculus.
    
    The chain rule says: if loss depends on A3 which depends on Z3
    which depends on W3, then:
        dLoss/dW3 = dLoss/dA3 * dA3/dZ3 * dZ3/dW3
    
    We compute this step by step, layer by layer, backwards.
    
    params: weights and biases
    cache: intermediate values from forward pass (Z1,A1,Z2,A2,Z3,A3)
    labels: one-hot encoded correct answers, shape (10, batch_size)
    
    Returns:
        grads: dictionary of gradients dW1,db1,dW2,db2,dW3,db3
    """
    grads = {}
    m = labels.shape[1]  # batch size — we average gradients over the batch

    # ── Layer 3 — output layer ─────────────────────────────────────────────
    # For softmax + cross entropy combined the gradient simplifies beautifully to:
    # dZ3 = predictions - true_labels
    # This is one of the elegant results in neural network math
    dZ3 = cache['A3'] - labels                          # (10, m)
    grads['dW3'] = (dZ3 @ cache['A2'].T) / m           # (10, 64)
    grads['db3'] = np.sum(dZ3, axis=1, keepdims=True) / m  # (10, 1)

    # ── Layer 2 — second hidden layer ──────────────────────────────────────
    # Chain rule: propagate gradient back through W3 then through ReLU
    dA2 = params['W3'].T @ dZ3                          # (64, m)
    dZ2 = dA2 * relu_derivative(cache['Z2'])            # (64, m)
    grads['dW2'] = (dZ2 @ cache['A1'].T) / m           # (64, 128)
    grads['db2'] = np.sum(dZ2, axis=1, keepdims=True) / m  # (64, 1)

    # ── Layer 1 — first hidden layer ───────────────────────────────────────
    # Same pattern — propagate back through W2 then through ReLU
    dA1 = params['W2'].T @ dZ2                          # (128, m)
    dZ1 = dA1 * relu_derivative(cache['Z1'])            # (128, m)
    grads['dW1'] = (dZ1 @ cache['A0'].T) / m           # (128, 784)
    grads['db1'] = np.sum(dZ1, axis=1, keepdims=True) / m  # (128, 1)

    return grads


def update_weights(params, grads, learning_rate):
    """
    Gradient descent — nudges every weight in the direction
    that reduces the loss.
    
    weight = weight - learning_rate * gradient
    
    Subtracting because the gradient points toward higher loss,
    we want to go the opposite direction toward lower loss.
    """
    num_layers = len(params) // 2  # W and b for each layer

    for l in range(1, num_layers + 1):
        params[f'W{l}'] -= learning_rate * grads[f'dW{l}']
        params[f'b{l}'] -= learning_rate * grads[f'db{l}']

    return params


if __name__ == "__main__":
    import numpy as np
    from network import initialise_weights, cross_entropy_loss
    from forward import forward_pass

    print("=== Backward Pass Test ===")

    np.random.seed(42)
    X_test = np.random.randn(784, 5)

    # Fake one-hot labels — pretend the answers are [3,7,3,8,8]
    labels = np.zeros((10, 5))
    labels[3, 0] = 1
    labels[7, 1] = 1
    labels[3, 2] = 1
    labels[8, 3] = 1
    labels[8, 4] = 1

    layer_sizes = [784, 128, 64, 10]
    params = initialise_weights(layer_sizes)

    # Forward pass
    output, cache = forward_pass(X_test, params)
    loss_before = cross_entropy_loss(output, labels)
    print(f"Loss before update: {loss_before:.4f}")

    # Backward pass
    grads = backward_pass(params, cache, labels)

    # Update weights
    params = update_weights(params, grads, learning_rate=0.01)

    # Forward pass again — loss should decrease
    output_after, _ = forward_pass(X_test, params)
    loss_after = cross_entropy_loss(output_after, labels)
    print(f"Loss after one update: {loss_after:.4f}")
    print(f"Loss decreased: {loss_before > loss_after}")

    print(f"\nGradient shapes (sanity check):")
    for key, val in grads.items():
        print(f"  {key}: {val.shape}")