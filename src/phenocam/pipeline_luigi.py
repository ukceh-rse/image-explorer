import glob
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List

import luigi
from skimage.io import imread, imsave

from phenocam.image.defisheye import do_defisheye
from phenocam.image.slice import slice_image_in_half

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
            try:
                image = imread(i)
            except OSError as err:
                logging.error(err)
                logging.info(i)
                continue

            left, right = slice_image_in_half(image)

            # Perform the defisheye operation

            defisheye_l = do_defisheye(left)
            # Save the defisheye image
            imsave(f"{self.output_directory}/{Path(i).stem}_L.jpg", defisheye_l)
            defisheye_r = do_defisheye(right)
            # Save the defisheye image
            # TODO consider resizing down to 256*256 (or 224) here, save storage
            try:
                imsave(f"{self.output_directory}/{Path(i).stem}_R.jpg", defisheye_r)
            except OSError as err:
                logging.error(err)
                logging.info(f"{Path(i).stem}_R.jpg")
        with self.output().open("w") as f:
            f.write("defisheye complete")


class PhenocamPipeline(luigi.WrapperTask):
    """
    Main wrapper task to execute the entire pipeline.
    """

    directory = luigi.Parameter()
    output_directory = luigi.Parameter()
    experiment_name = luigi.Parameter()

    def requires(self) -> luigi.Task:
        return DefisheyeImages(
            directory=self.directory,
            output_directory=self.output_directory,
            experiment_name=self.experiment_name,
        )


# To run the pipeline
if __name__ == "__main__":
    luigi.run(
        [
            "PhenocamPipeline",
            # "--local-scheduler",
            "--directory",
            "./tests/fixtures",
            "--output-directory",
            "./data/images_decollage",
            "--experiment-name",
            "test",
        ]
    )
