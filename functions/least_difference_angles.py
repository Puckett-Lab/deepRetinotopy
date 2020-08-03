import numpy as np


def smallest_angle(x, y):
    """Least difference between two angles

    Args:
      x (numpy array): An array of shape (number of vertices,1) containing
      the empirical polar angles
      y (numpy array): An array of shape (number of vertices,1) containing
      the predicted polar angles

    Returns:
      numpy array: the difference between predicted and empirical polar angles
    """
    difference = []
    dif_1 = np.abs(y - x)
    dif_2 = np.abs(y - x + 2 * np.pi)
    dif_3 = np.abs(y - x - 2 * np.pi)
    for i in range(len(x)):
        difference.append(min(dif_1[i], dif_2[i], dif_3[i]))
    return np.array(difference) * 180 / np.pi
