"""Notebook code, bletherily refactored by CoPilot"""

import glob
import logging
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
from thingsvision import get_extractor
from thingsvision.core.extraction.torch import PyTorchExtractor
from thingsvision.utils.data import DataLoader, ImageDataset
from thingsvision.utils.storing import save_features


def file_listing(directory: str, suffix: Optional[str] = "jpg") -> List[str]:
    """
    Return a listing of files matching `suffix` in a directory.

    :param directory: Directory to search for files.
    :type directory: str
    :param suffix: File suffix to match, defaults to "jpg".
    :type suffix: Optional[str]
    :return: List of filenames matching the suffix.
    :rtype: List[str]
    """
    return [os.path.basename(x) for x in glob.glob(f"{directory}/*.{suffix}")]


def file_metadata(filename: str) -> Tuple[str, datetime]:
    """
    Parse a PhenoCam filename into site name and datetime.

    :param filename: Image filename in format SITE_YYYYMMDD_HHMMSS_ID.jpg.
    :type filename: str
    :return: A tuple containing site name and timestamp of the image.
    :rtype: Tuple[str, datetime]
    :raises ValueError: If filename format is invalid.
    """
    try:
        # The last value should be north-south orientation
        site, date, time, _, _ = filename.split("_")
    except ValueError as err:
        logging.error(filename)
        logging.error("filename needs to be SITE_YYYYMMDD_HHMMSS_ID_ORIENTATION.jpg")
        logging.error(err)
        raise
    dt = datetime.strptime(date + time, "%Y%m%d%H%M%S")
    return site, dt


def create_dataset(root: str, out_path: str, extractor: PyTorchExtractor) -> ImageDataset:
    """
    Create an ImageDataset object from a directory of images.

    :param root: Root directory containing input images.
    :type root: str
    :param out_path: Output directory path for processed features.
    :type out_path: str
    :param extractor: Feature extractor instance.
    :type extractor: PyTorchExtractor
    :return: Dataset object configured with the provided extractor.
    :rtype: ImageDataset
    """
    return ImageDataset(
        root=root,
        out_path=out_path,
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )


def create_dataloader(dataset: ImageDataset, batch_size: int, extractor: PyTorchExtractor) -> DataLoader:
    """
    Create a PyTorch DataLoader for batch processing.

    :param dataset: Input dataset.
    :type dataset: ImageDataset
    :param batch_size: Number of samples per batch.
    :type batch_size: int
    :param extractor: Feature extractor instance.
    :type extractor: PyTorchExtractor
    :return: Configured PyTorch DataLoader instance.
    :rtype: DataLoader
    """
    return DataLoader(dataset=dataset, batch_size=batch_size, backend=extractor.get_backend())


def extract_features(
    extractor: PyTorchExtractor, dataloader: DataLoader, module_name: str, flatten_acts: bool = True
) -> np.ndarray:
    """
    Extract features from the dataset using the specified module of the extractor.

    :param extractor: Feature extractor instance.
    :type extractor: PyTorchExtractor
    :param dataloader: DataLoader for batch processing.
    :type dataloader: DataLoader
    :param module_name: Name of the module to extract features from.
    :type module_name: str
    :param flatten_acts: Whether to flatten the activation maps, defaults to True.
    :type flatten_acts: bool
    :return: Extracted features.
    :rtype: np.ndarray
    """
    features = extractor.extract_features(batches=dataloader, module_name=module_name, flatten_acts=flatten_acts)
    return features


def main() -> None:
    """
    Main function to execute the feature extraction process.
    """
    model_name = "simclr-rn50"
    source = "ssl"
    device = "cpu"
    root = "./tests/sample"
    out_path = "./features"
    batch_size = 16
    module_name = "fc"

    extractor = get_extractor(model_name=model_name, source=source, device=device, pretrained=True)
    dataset = create_dataset(root, out_path, extractor)
    dataloader = create_dataloader(dataset, batch_size, extractor)
    save_features(extract_features(extractor, dataloader, module_name), out_path=out_path, file_format="npy")


if __name__ == "__main__":
    main()
