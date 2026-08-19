import numpy as np
import matplotlib.pyplot as plt
from network import initialise_weights, cross_entropy_loss
from forward import forward_pass
from backward import backward_pass, update_weights
from data import load_mnist
from train import train, get_accuracy


def show_predictions(X_test, Y_test, params, n=10):
    """
    Shows n random test images with the network's prediction
    and confidence. Green = correct, red = wrong.
    """
    indices = np.random.choice(X_test.shape[1], n, replace=False)
    X_sample = X_test[:, indices]
    Y_sample = Y_test[:, indices]

    predictions, _ = forward_pass(X_sample, params)
    predicted_digits = np.argmax(predictions, axis=0)
    true_digits = np.argmax(Y_sample, axis=0)
    confidences = np.max(predictions, axis=0)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Network Predictions on Test Images', fontsize=13)

    for i, ax in enumerate(axes.flat):
        image = X_sample[:, i].reshape(28, 28)
        predicted = predicted_digits[i]
        true = true_digits[i]
        confidence = confidences[i]
        correct = predicted == true

        ax.imshow(image, cmap='gray')
        ax.axis('off')
        color = 'green' if correct else 'red'
        ax.set_title(
            f"Pred: {predicted} ({confidence*100:.0f}%)\nTrue: {true}",
            color=color, fontsize=9
        )

    plt.tight_layout()
    plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved predictions.png")


def show_misclassified(X_test, Y_test, params, n=10):
    """
    Shows examples the network got wrong.
    These are the most interesting cases — often genuinely ambiguous.
    """
    predictions, _ = forward_pass(X_test, params)
    predicted = np.argmax(predictions, axis=0)
    true = np.argmax(Y_test, axis=0)
    wrong = np.where(predicted != true)[0]

    print(f"\nTotal misclassified: {len(wrong)} / {X_test.shape[1]}")

    sample = wrong[:n]
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Misclassified Examples', fontsize=13)

    for i, ax in enumerate(axes.flat):
        idx = sample[i]
        image = X_test[:, idx].reshape(28, 28)
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        confidence = np.max(predictions[:, idx])
        ax.set_title(
            f"Pred: {predicted[idx]} ({confidence*100:.0f}%)\nTrue: {true[idx]}",
            color='red', fontsize=9
        )

    plt.tight_layout()
    plt.savefig('misclassified.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved misclassified.png")


def show_confusion_matrix(X_test, Y_test, params):
    """
    Confusion matrix — shows which digits get confused with which.
    Row = true digit, Column = predicted digit.
    Perfect classifier = diagonal only.
    """
    predictions, _ = forward_pass(X_test, params)
    predicted = np.argmax(predictions, axis=0)
    true = np.argmax(Y_test, axis=0)

    matrix = np.zeros((10, 10), dtype=int)
    for t, p in zip(true, predicted):
        matrix[t][p] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap='Blues')

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xlabel('Predicted digit', fontsize=11)
    ax.set_ylabel('True digit', fontsize=11)
    ax.set_title('Confusion Matrix', fontsize=13)

    for i in range(10):
        for j in range(10):
            color = 'white' if matrix[i, j] > 500 else 'black'
            ax.text(j, i, str(matrix[i, j]),
                   ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    print("Loading MNIST and training...")
    X_train, Y_train, X_test, Y_test = load_mnist()

    params, losses, accuracies = train(
        X_train, Y_train, X_test, Y_test,
        epochs=20, learning_rate=0.1, batch_size=256
    )

    print("\nGenerating visualisations...")
    show_predictions(X_test, Y_test, params)
    show_misclassified(X_test, Y_test, params)
    show_confusion_matrix(X_test, Y_test, params)