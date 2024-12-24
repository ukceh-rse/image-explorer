import pytest
import numpy as np
import os


@pytest.fixture
def fixture_dir():
    """
    Base directory for the test fixtures (images, metadata)
    """
    return os.path.join(os.path.abspath(os.path.dirname(__file__)), "fixtures/")


@pytest.fixture
def sample_image():
    # Create 400x400 RGB image
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    y, x = np.ogrid[-200:200, -200:200]
    r = np.sqrt(x * x + y * y)

    # Create concentric circles with different RGB colors
    colors = [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
    ]

    for i, color in enumerate(colors):
        radius = 40 * (i + 1)
        mask = (r >= radius - 20) & (r < radius)
        image[mask] = color

    return image


@pytest.fixture
def sample_wide_image():
    # Create 100x200x3 image with recognizable pattern
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    # Left half red
    img[:, :100, 0] = 255
    # Right half blue
    img[:, 100:, 2] = 255
    return img
