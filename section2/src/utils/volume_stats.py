"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b, eps=1e-6):
    """
    Compute Dice Similarity coefficient for two 3D volumes.
    Volumes are expected to be the same size. 
    We treat 0 as background, anything else as foreground.

    Arguments:
        a {Numpy array} -- 3D array (prediction)
        b {Numpy array} -- 3D array (ground truth)

    Returns:
        float -- Dice coefficient in [0, 1]
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # Convert to binary (foreground vs background)
    a_bin = (a > 0).astype(np.int32)
    b_bin = (b > 0).astype(np.int32)

    intersection = np.sum(a_bin * b_bin)
    return (2. * intersection) / (np.sum(a_bin) + np.sum(b_bin) + eps)


def Jaccard3d(a, b, eps=1e-6):
    """
    Compute Jaccard Similarity coefficient (a.k.a. IoU) for two 3D volumes.
    Volumes are expected to be the same size. 
    We treat 0 as background, anything else as foreground.

    Arguments:
        a {Numpy array} -- 3D array (prediction)
        b {Numpy array} -- 3D array (ground truth)

    Returns:
        float -- Jaccard index in [0, 1]
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    a_bin = (a > 0).astype(np.int32)
    b_bin = (b > 0).astype(np.int32)

    intersection = np.sum(a_bin * b_bin)
    union = np.sum(a_bin) + np.sum(b_bin) - intersection
    return intersection / (union + eps)
