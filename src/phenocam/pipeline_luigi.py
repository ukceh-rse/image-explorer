import glob
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

import luigi
import numpy as np
from skimage.io import imread

from phenocam.data.vectorstore import SQLiteVecStore, serialize_f32, vector_store
from phenocam.features.utils import (
    create_dataloader,
    create_dataset,
    extract_features,
    file_metadata,
    get_extractor,
    save_features,
)
from phenocam.image.defisheye import do_defisheye
from phenocam.image.slice import save_image, slice_image_in_half

# Set up logging
logging.basicConfig(level=logging.INFO)


class CreateOutputDirectory(luigi.Task):
    """
    Task to create the output directory if it does not exist.
    """

    output_directory = luigi.Parameter()

    def output(self) -> luigi.Target:
        return luigi.LocalTarget(self.output_directory)

    def run(self) -> None:
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)
            logging.info(f"Output directory created: {self.output_directory}")
        else:
            logging.info(f"Output directory already exists: {self.output_directory}")


class DefisheyeImages(luigi.Task):
    """
    Task that processes the large TIFF image, extracts vignettes, and saves them with EXIF metadata.
    """

    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()

    def requires(self) -> List[luigi.Task]:
        return [CreateOutputDirectory(self.output_directory)]

    def output(self) -> luigi.Target:
        date = datetime.today().date()
        return luigi.LocalTarget(f"{self.directory}/defisheye_complete_{date}.txt")

    def run(self) -> None:
        # Load the image
        image_files = glob.glob(f"{self.directory}/*.jpg")
        for i in image_files:
            stem = Path(i).stem
            try:
                image = imread(i)
            except OSError as err:
                logging.error(err)
                logging.info(i)
                continue

            left, right = slice_image_in_half(image)
            save_image(self.output_directory, stem, "L", do_defisheye(left))
            save_image(self.output_directory, stem, "R", do_defisheye(right))

        with self.output().open("w") as f:
            f.write("defisheye complete")


class ExtractEmbeddings(luigi.Task):
    model_name = "simclr-rn50"
    source = "ssl"
    device = "cpu"
    batch_size = 16
    module_name = "layer4.2.conv3"

    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()
    data_directory = luigi.Parameter()

    def requires(self) -> List[luigi.Task]:
        return [
            DefisheyeImages(
                directory=self.directory,
                output_directory=self.output_directory,
                experiment_name=self.experiment_name,
            )
        ]

    def output(self) -> luigi.Target:
        date = datetime.today().date()
        return luigi.LocalTarget(f"{self.data_directory}/embeddings_complete_{date}.txt")

    def run(self) -> None:
        extractor = get_extractor(model_name=self.model_name, source=self.source, device=self.device, pretrained=True)
        dataset = create_dataset(self.directory, extractor)
        dataloader = create_dataloader(dataset, self.batch_size, extractor)
        try:
            save_features(
                extract_features(extractor, dataloader, self.module_name),
                out_path=self.data_directory,
                file_format="npy",
            )
            with self.output().open("w") as f:
                f.write(f"{self.data_directory}/file_names.txt")
            pass
        except Exception as e:
            logging.error(e)


class SaveMetadata(luigi.Task):
    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()
    data_directory = luigi.Parameter()

    def requires(self) -> List[luigi.Task]:
        return ExtractEmbeddings(
            directory=self.directory,
            output_directory=self.output_directory,
            experiment_name=self.experiment_name,
            data_directory=self.data_directory,
        )

    def store(self) -> SQLiteVecStore:
        return vector_store("sqlite", f"{self.data_directory}/{self.experiment_name}.db")

    def output(self) -> luigi.Target:
        date = datetime.today().date()
        return luigi.LocalTarget(f"{self.data_directory}/metadata_complete_{date}.txt")

    def run(self) -> None:
        with open(f"{self.data_directory}/file_names.txt") as f:
            file_names = f.readlines()

        feature_map = []
        with open(f"{self.data_directory}/features.npy") as f:
            feature_map = np.load(f"{self.data_directory}/features.npy")

        try:
            for index, name in enumerate(file_names):
                site, dt = file_metadata(name)
                # We could make these full HTTPs URLs - think about storage structure
                self.store().add(url=name, embeddings=serialize_f32(feature_map[index]), date=dt, site=site)

            with self.output().open("w") as f:
                f.write(f"{self.data_directory}/file_names.txt")
        except Exception as err:
            logging.error(err)
            raise


class PhenocamPipeline(luigi.WrapperTask):
    """
    Main wrapper task to execute the entire pipeline.
    """

    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()
    data_directory = luigi.Parameter()

    def requires(self) -> luigi.Task:
        return SaveMetadata(
            directory=self.directory,
            output_directory=self.output_directory,
            experiment_name=self.experiment_name,
            data_directory=self.data_directory,
        )


# To run the pipeline
if __name__ == "__main__":
    luigi.run(
        [
            "PhenocamPipeline",
            # "--local-scheduler",
            "--directory",
            # "./tests/fixtures/",
            "../Phenocam_samples/2024/WADDN/",
            "--output-directory",
            "./data/images_decollage",
            "--experiment-name",
            "test",
            "--data-directory",
            "./data/vectors",
        ]
    )
