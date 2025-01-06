"""Notebook code, refactored by CoPilot"""

from typing import Any

from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader, ImageDataset


def create_extractor(model_name: str, source: str, device: str, pretrained: bool = True) -> Any:
    return get_extractor(model_name=model_name, source=source, device=device, pretrained=pretrained)


def create_dataset(root: str, extractor: Any) -> ImageDataset:
    return ImageDataset(
        root=root,
        out_path="../features",
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )


def create_dataloader(dataset: ImageDataset, batch_size: int, extractor: Any) -> DataLoader:
    return DataLoader(dataset=dataset, batch_size=batch_size, backend=extractor.get_backend())


def extract_features(extractor: Any, dataloader: DataLoader, module_name: str, flatten_acts: bool = True) -> None:
    features = extractor.extract_features(batches=dataloader, module_name=module_name, flatten_acts=flatten_acts)
    return features


def main() -> None:
    model_name = "simclr-rn50"
    source = "ssl"
    device = "cpu"
    root = "./tests/fixtures"
    batch_size = 1
    module_name = "layer4.2.conv3"

    extractor = create_extractor(model_name, source, device)
    dataset = create_dataset(root, extractor)
    dataloader = create_dataloader(dataset, batch_size, extractor)
    extract_features(extractor, dataloader, module_name)


if __name__ == "__main__":
    main()
