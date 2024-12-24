from typing import Tuple

import numpy as np


def slice_image_in_half(image_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slices an image into two halves.

    Args:
      image_array: A NumPy array representing the image.

    Returns:
      A tuple containing two NumPy arrays representing the left and right halves of the image.
    """
    _, width, _ = image_array.shape
    midpoint = width // 2
    left_half = image_array[:, :midpoint, :]
    right_half = image_array[:, midpoint:, :]
    return left_half, right_half
