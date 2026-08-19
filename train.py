import numpy as np
import matplotlib.pyplot as plt
from network import initialise_weights, cross_entropy_loss
from forward import forward_pass
from backward import backward_pass, update_weights
from data import load_mnist


def get_accuracy(predictions, labels):
    """
    Compares predicted digits to true digits.
    argmax gets the index of the highest probability — that's the prediction.
    """
    predicted = np.argmax(predictions, axis=0)
    true      = np.argmax(labels, axis=0)
    return np.mean(predicted == true)


def get_batch(X, Y, batch_size):
    """
    Returns a random mini-batch of examples.
    
    Why mini-batches instead of the full dataset?
    - Full dataset: accurate gradients but slow and memory heavy
    - Single example: fast but very noisy gradients
    - Mini-batch: best of both — fast enough, stable enough
    Typical sizes: 32, 64, 128, 256
    """
    m = X.shape[1]
    indices = np.random.choice(m, batch_size, replace=False)
    return X[:, indices], Y[:, indices]


def train(X_train, Y_train, X_test, Y_test,
          layer_sizes=[784, 128, 64, 10],
          learning_rate=0.1,
          epochs=20,
          batch_size=256):
    """
    Full training loop.
    
    Each epoch:
    1. Shuffle and split training data into mini-batches
    2. For each batch: forward pass → loss → backward pass → update weights
    3. After all batches: evaluate accuracy on test set
    4. Print progress
    
    epoch: one full pass through the training data
    """
    params = initialise_weights(layer_sizes)

    train_losses = []
    test_accuracies = []

    print(f"Training network {layer_sizes}")
    print(f"Learning rate: {learning_rate} | Epochs: {epochs} | Batch size: {batch_size}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        epoch_losses = []
        m = X_train.shape[1]
        num_batches = m // batch_size

        # Shuffle training data each epoch
        indices = np.random.permutation(m)
        X_shuffled = X_train[:, indices]
        Y_shuffled = Y_train[:, indices]

        for b in range(num_batches):
            # Get mini-batch
            start = b * batch_size
            end   = start + batch_size
            X_batch = X_shuffled[:, start:end]
            Y_batch = Y_shuffled[:, start:end]

            # Forward pass
            predictions, cache = forward_pass(X_batch, params)

            # Compute loss
            loss = cross_entropy_loss(predictions, Y_batch)
            epoch_losses.append(loss)

            # Backward pass
            grads = backward_pass(params, cache, Y_batch)

            # Update weights
            params = update_weights(params, grads, learning_rate)

        # Evaluate on test set after each epoch
        test_preds, _ = forward_pass(X_test, params)
        test_acc = get_accuracy(test_preds, Y_test)

        # Training accuracy on a sample
        train_preds, _ = forward_pass(X_train[:, :1000], params)
        train_acc = get_accuracy(train_preds, Y_train[:, :1000])

        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch:2d}/{epochs} | "
              f"Loss: {avg_loss:.4f} | "
              f"Train acc: {train_acc*100:.1f}% | "
              f"Test acc: {test_acc*100:.1f}%")

    return params, train_losses, test_accuracies


def plot_results(train_losses, test_accuracies):
    """
    Plots training loss and test accuracy over epochs.
    This is how you diagnose whether training is working:
    - Loss should decrease
    - Accuracy should increase
    - If loss decreases but accuracy doesn't — something is wrong
    - If both plateau early — try a different learning rate
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, color='royalblue', linewidth=2)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in test_accuracies], color='seagreen', linewidth=2)
    ax2.set_title('Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved as training_results.png")


if __name__ == "__main__":
    print("Loading MNIST...")
    X_train, Y_train, X_test, Y_test = load_mnist()
    print(f"Loaded {X_train.shape[1]} training, {X_test.shape[1]} test images\n")

    params, losses, accuracies = train(
        X_train, Y_train, X_test, Y_test,
        layer_sizes   = [784, 128, 64, 10],
        learning_rate = 0.1,
        epochs        = 20,
        batch_size    = 256
    )

    print(f"\nFinal test accuracy: {accuracies[-1]*100:.2f}%")
    plot_results(losses, accuracies)