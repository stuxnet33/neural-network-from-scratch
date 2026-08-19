import numpy as np
from network import relu, softmax, initialise_weights

def forward_pass(X, params):
    """
    Passes input data through every layer of the network.
    
    X: input data, shape (784, batch_size)
       each column is one image flattened to 784 pixels
    params: dictionary of weights and biases W1,b1,W2,b2,W3,b3
    
    For each layer we do two things:
    Z = W @ X + b   — linear transformation (matrix multiply + bias)
    A = activation(Z) — non-linear activation
    
    We store all intermediate values in 'cache' because
    backpropagation needs them to compute gradients.
    
    Returns:
        output: softmax probabilities shape (10, batch_size)
        cache: all intermediate Z and A values
    """
    cache = {}

    # Layer 1 — input to first hidden layer
    # Z1 shape: (128, batch_size)
    cache['A0'] = X
    cache['Z1'] = params['W1'] @ X + params['b1']
    cache['A1'] = relu(cache['Z1'])

    # Layer 2 — first hidden to second hidden
    # Z2 shape: (64, batch_size)
    cache['Z2'] = params['W2'] @ cache['A1'] + params['b2']
    cache['A2'] = relu(cache['Z2'])

    # Layer 3 — second hidden to output
    # Z3 shape: (10, batch_size)
    cache['Z3'] = params['W3'] @ cache['A2'] + params['b3']
    cache['A3'] = softmax(cache['Z3'])  # output layer uses softmax not relu

    return cache['A3'], cache


if __name__ == "__main__":
    print("=== Forward Pass Test ===")

    # Simulate a batch of 5 images — random pixels
    # In reality this would be MNIST images
    np.random.seed(42)
    X_test = np.random.randn(784, 5)  # 5 images, 784 pixels each

    # Initialise random weights
    layer_sizes = [784, 128, 64, 10]
    params = initialise_weights(layer_sizes)

    # Run forward pass
    output, cache = forward_pass(X_test, params)

    print(f"Input shape:  {X_test.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nPredictions for 5 images (rows=digits 0-9, cols=images):")
    print(np.round(output, 3))
    print(f"\nProbabilities sum to 1 for each image: {np.round(np.sum(output, axis=0), 4)}")
    print(f"\nPredicted digit for each image: {np.argmax(output, axis=0)}")