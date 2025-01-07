"""Notebook code, bletherily refactored by CoPilot"""

import glob
import logging
import os
from datetime import datetime
from typing import List, Optional

import numpy as np
from thingsvision import get_extractor
from thingsvision.core.extraction.torch import PyTorchExtractor
from thingsvision.utils.data import DataLoader, ImageDataset
from thingsvision.utils.storing import save_features


def file_listing(directory: str, suffix: Optional[str] = "jpg") -> List[str]:
    """Return a listing of files matching `suffix` in a directory
    This was input to file_names for ImageDataset, but that class only writes the list without it
    Left here for now in case it's needed later
    """
    return [os.path.basename(x) for x in glob.glob(f"{directory}/*.{suffix}")]


def file_metadata(filename: str) -> tuple:
    """Return a tuple of metadata for a file"""
    # The last field is an identifier, inconsistent across years, we probably want to keep it
    try:
        site, date, time, _ = filename.split("_")
    except ValueError as err:
        logging.error("filename needs to be SITE_DAY_TIME_ID.jpg")
        logging.error(err)
        raise
    dt = datetime.strptime(date + time, "%Y%m%d%H%M%S")
    return site, dt


def create_dataset(root: str, extractor: PyTorchExtractor) -> ImageDataset:
    """Create an ImageDataset object from a directory of images"""
    return ImageDataset(
        root=root,
        out_path="./features",
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )


def create_dataloader(dataset: ImageDataset, batch_size: int, extractor: PyTorchExtractor) -> DataLoader:
    """Create a pytorch DataLoader object from an ImageDataset"""
    return DataLoader(dataset=dataset, batch_size=batch_size, backend=extractor.get_backend())


def extract_features(
    extractor: PyTorchExtractor, dataloader: DataLoader, module_name: str, flatten_acts: bool = True
) -> np.ndarray:
    features = extractor.extract_features(batches=dataloader, module_name=module_name, flatten_acts=flatten_acts)
    return features


def main() -> None:
    model_name = "simclr-rn50"
    source = "ssl"
    device = "cpu"
    root = "./tests/sample"
    batch_size = 16
    module_name = "fc"

    extractor = get_extractor(model_name=model_name, source=source, device=device, pretrained=True)
    dataset = create_dataset(root, extractor)
    dataloader = create_dataloader(dataset, batch_size, extractor)
    save_features(extract_features(extractor, dataloader, module_name), out_path="features", file_format="npy")


if __name__ == "__main__":
    main()
