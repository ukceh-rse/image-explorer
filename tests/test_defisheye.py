import numpy as np
from imagesearch.image.defisheye import do_defisheye


def test_do_defisheye_returns_ndarray(sample_image):
    a = do_defisheye(sample_image)
    assert isinstance(a, np.ndarray)


def test_do_defisheye_preserves_dimensions(sample_image):
    a = do_defisheye(sample_image)
    assert a.shape == sample_image.shape
