from phenocam.features.utils import file_metadata
from datetime import datetime
import pytest

def test_file_metadata():
    # Are there other consistent file naming conventions to look out for?
    site, dt = file_metadata("WADDN_20140101_0902_ID405.jpg")
    assert site == 'WADDN'
    assert dt == datetime(2014, 1, 1, 9, 0, 2)

    site, dt = file_metadata("WADDN_20240101_0910_ID20240101091001.jpg")
    assert dt == datetime(2024, 1, 1, 9, 1)

    with pytest.raises(ValueError):
        _, _ = file_metadata("NOPE_2025.jpg")

