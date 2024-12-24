import numpy as np
from defisheye import Defisheye


def do_defisheye(image: np.ndarray) -> np.ndarray:
    """
    Run the "defisheye" method on a hemisphere image with reasonable defaults.
    """
    dtype = "linear"
    img_format = "fullframe"
    fov = 180
    pfov = 120

    obj = Defisheye(image, dtype=dtype, format=img_format, fov=fov, pfov=pfov)

    return obj.convert()
