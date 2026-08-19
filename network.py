import numpy as np

# ── Activation functions ───────────────────────────────────────────────────────

def relu(z):
    """
    ReLU — Rectified Linear Unit.
    If input is positive, pass it through. If negative, return 0.
    This introduces non-linearity — without it the whole network
    collapses into a single linear equation no matter how many layers.
    """
    return np.maximum(0, z)

def relu_derivative(z):
    """
    Derivative of ReLU — used during backpropagation.
    1 where input was positive, 0 where it was negative.
    Tells us how much the activation changed relative to its input.
    """
    return (z > 0).astype(float)

def softmax(z):
    """
    Softmax — converts raw scores into probabilities that sum to 1.
    Used on the output layer for classification.
    
    We subtract the max for numerical stability — prevents overflow
    when exponentiating large numbers. Mathematically equivalent
    but numerically safer.
    """
    z_stable = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# ── Loss function ──────────────────────────────────────────────────────────────

def cross_entropy_loss(predictions, labels):
    """
    Cross-entropy loss for multi-class classification.
    
    predictions: softmax output, shape (num_classes, batch_size)
    labels: one-hot encoded, shape (num_classes, batch_size)
    
    -log(probability assigned to correct class)
    High when confident and wrong, low when confident and right.
    """
    m = labels.shape[1]  # batch size
    # Clip predictions to avoid log(0) which is undefined
    predictions_clipped = np.clip(predictions, 1e-15, 1 - 1e-15)
    loss = -np.sum(labels * np.log(predictions_clipped)) / m
    return loss


# ── Weight initialisation ──────────────────────────────────────────────────────

def initialise_weights(layer_sizes):
    """
    Initialises weights and biases for all layers.
    
    layer_sizes: list of integers e.g. [784, 128, 64, 10]
    means: input layer 784 neurons, two hidden layers, output 10 neurons
    
    We use He initialisation — scale weights by sqrt(2/n_inputs).
    This prevents vanishing/exploding gradients in deep networks.
    Random small weights break symmetry — if all weights were equal,
    every neuron would learn the same thing and the network can't specialise.
    
    Returns dict with W1, b1, W2, b2 etc for each layer.
    """
    params = {}
    num_layers = len(layer_sizes) - 1

    for l in range(1, num_layers + 1):
        n_in  = layer_sizes[l - 1]
        n_out = layer_sizes[l]

        # He initialisation — good default for ReLU networks
        params[f'W{l}'] = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
        params[f'b{l}'] = np.zeros((n_out, 1))

    return params


# ── Quick sanity check ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Sanity checks ===")

    # ReLU test
    z = np.array([-2, -1, 0, 1, 2], dtype=float)
    print(f"ReLU({z}) = {relu(z)}")
    print(f"ReLU derivative({z}) = {relu_derivative(z)}")

    # Softmax test — should sum to 1
    scores = np.array([[2.0], [1.0], [0.5]])
    probs = softmax(scores)
    print(f"\nSoftmax({scores.T}) = {probs.T}")
    print(f"Sum of probabilities: {np.sum(probs):.4f} (should be 1.0)")

    # Weight initialisation test
    layer_sizes = [784, 128, 64, 10]
    params = initialise_weights(layer_sizes)
    print(f"\nWeight shapes for network {layer_sizes}:")
    for key, val in params.items():
        print(f"  {key}: {val.shape}")