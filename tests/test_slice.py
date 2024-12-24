from phenocam.image.slice import slice_image_in_half
import numpy as np


def test_slice_returns_tuple(sample_wide_image):
    result = slice_image_in_half(sample_wide_image)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_slice_dimensions(sample_wide_image):
    left, right = slice_image_in_half(sample_wide_image)
    height, full_width, channels = sample_wide_image.shape
    expected_width = full_width // 2

    assert left.shape == (height, expected_width, channels)
    assert right.shape == (height, expected_width, channels)


def test_slice_content(sample_wide_image):
    left, right = slice_image_in_half(sample_wide_image)
    # Left half should be red
    assert np.all(left[:, :, 0] == 255)  # Red channel
    assert np.all(left[:, :, 1:] == 0)  # Green and Blue channels
    # Right half should be blue
    assert np.all(right[:, :, 2] == 255)  # Blue channel
    assert np.all(right[:, :, :2] == 0)  # Red and Green channels


def test_slice_odd_width():
    # Create 100x201x3 image (odd width)
    odd_image = np.zeros((100, 201, 3), dtype=np.uint8)
    left, right = slice_image_in_half(odd_image)
    assert left.shape[1] + right.shape[1] == odd_image.shape[1]
