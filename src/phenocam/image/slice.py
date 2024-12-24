from typing import Optional, Tuple

import numpy as np
from PIL import Image


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


def save_image(directory: str, file: str, half: str, image: np.ndarray, size: Optional[int] = 600) -> None:
    """
    Saves the left and right halves of an image to disk.

    Args:
      directory: The directory where the halves should be saved.
      half: The half of the image to save ('L' for left, 'R' for right).
      file: The base filename of the image.
      image: A NumPy array representing the image.
      size: The size to which the image should be resized. Defaults to 600.
    """

    filename = f"{directory}/{file}_{half}.jpg"

    # Resize image to 600x600 if not already to match 2014
    # Done with PIL because skimage will force cast to float even with preserve_range=True
    img = Image.fromarray(image)
    if img.size[0] != size:
        img = img.resize((size, size))

    img.save(filename)
    return filename
