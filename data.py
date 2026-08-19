import numpy as np
import urllib.request
import gzip
import os

def download_mnist():
    """
    Downloads the MNIST dataset directly from the source.
    MNIST is 70,000 handwritten digit images (28x28 pixels).
    60,000 for training, 10,000 for testing.
    """
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]

    os.makedirs("data", exist_ok=True)

    for file in files:
        path = f"data/{file}"
        if not os.path.exists(path):
            print(f"Downloading {file}...")
            urllib.request.urlretrieve(base_url + file, path)
        else:
            print(f"Already have {file}")


def load_mnist():
    """
    Loads and preprocesses MNIST data.
    
    Preprocessing:
    - Pixel values normalised from 0-255 to 0-1
      (smaller values = more stable gradients)
    - Images flattened from 28x28 to 784-element vectors
    - Labels converted to one-hot encoding
      e.g. digit 3 becomes [0,0,0,1,0,0,0,0,0,0]
    
    Returns:
        X_train: (784, 60000) — training images
        Y_train: (10, 60000)  — training labels one-hot
        X_test:  (784, 10000) — test images
        Y_test:  (10, 10000)  — test labels one-hot
    """
    download_mnist()

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            f.read(16)  # skip header
            data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(-1, 784).T / 255.0  # normalise and transpose

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            f.read(8)  # skip header
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        # Convert to one-hot
        one_hot = np.zeros((10, len(labels)))
        one_hot[labels, np.arange(len(labels))] = 1
        return one_hot

    X_train = read_images("data/train-images-idx3-ubyte.gz")
    Y_train = read_labels("data/train-labels-idx1-ubyte.gz")
    X_test  = read_images("data/t10k-images-idx3-ubyte.gz")
    Y_test  = read_labels("data/t10k-labels-idx1-ubyte.gz")

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    print("Loading MNIST...")
    X_train, Y_train, X_test, Y_test = load_mnist()

    print(f"\nDataset shapes:")
    print(f"  X_train: {X_train.shape}  — 60,000 training images")
    print(f"  Y_train: {Y_train.shape}  — 60,000 training labels")
    print(f"  X_test:  {X_test.shape}   — 10,000 test images")
    print(f"  Y_test:  {Y_test.shape}   — 10,000 test labels")

    print(f"\nPixel value range: {X_train.min():.1f} to {X_train.max():.1f}")
    print(f"Label for first training image: {np.argmax(Y_train[:, 0])}")
    print(f"One-hot encoding: {Y_train[:, 0].astype(int)}")