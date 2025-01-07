"""
Streamlit application to visualise how plankton cluster
based on their embeddings from a deep learning model

* Metadata in intake catalogue (basically a dataframe of filenames
  - later this could have lon/lat, date, depth read from Exif headers
* Embeddings in chromadb, linked by filename

"""

import random
from typing import Optional

import streamlit as st
from PIL import Image

from phenocam.data.db_config import OPTIONS
from phenocam.data.vectorstore import vector_store

# KLUDGE - not reproducible at all
# Depends on a mirror of Phenocam images in /YYYY/SITE/ layout
# with http server that's python -m http.server
IMAGE_BASEURL = "https://localhost:8000"


@st.cache_resource
def store(coll: str) -> None:
    """
    Load the vector store with image embeddings.
    """
    # TODO stop recreating the connection on every call
    # E.g. chroma will have one store per collection...

    return vector_store("sqlite", coll, **OPTIONS["sqlite"])


@st.cache_data
def image_ids(coll: str) -> list:
    """
    Retrieve image embeddings from chroma database.
    TODO Revisit our available metadata
    """
    return store(coll).ids()


@st.cache_data
def image_embeddings() -> list:
    return store(st.session_state["collection"]).embeddings()


def closest_n(url: str, n: Optional[int] = 26) -> list:
    """
    Given an image URL return the N closest ones by cosine distance
    """
    s = store(st.session_state["collection"])
    embed = s.get(url)
    results = s.closest(embed, n_results=n)
    return results


@st.cache_data
def cached_image_url(filename: str) -> Image:
    """
    Convert an image filename into a HTTPS URL and return it
    Short-term kludge while we only have a local store of images (for good reason)
    Medium-term is to read from s3, authenticated, with requests and return BytesIO
    Longer-term is go back to doing this but with a stable URL scheme
    """
    # parts = filename.split("_")
    # site = parts[0]
    # year = parts[1][:4]
    # return f"{IMAGE_BASEURL}/{year}/{site}/{filename}"
    return f"{IMAGE_BASEURL}/{filename}"


def closest_grid(start_url: str, size: Optional[int] = 65) -> None:
    """
    Given an image URL, render a grid of the N nearest images
    by cosine distance between embeddings
    N defaults to 26
    """
    closest = closest_n(start_url, size)

    # TODO understand where layout should happen
    rows = []
    for _ in range(0, 8):
        rows.append(st.columns(8))

    for index, _ in enumerate(rows):
        for c in rows[index]:
            try:
                next_image = closest.pop()
            except IndexError:
                break
            c.image(cached_image_url(next_image), width=60)
            c.button("this", key=next_image, on_click=pick_image, args=[next_image])


def random_image() -> str:
    ids = image_ids(st.session_state["collection"])
    # starting image
    test_image_url = random.choice(ids)
    return test_image_url


def pick_image(image: str) -> None:
    st.session_state["random_img"] = image


def show_random_image() -> None:
    if st.session_state["random_img"]:
        st.image(cached_image_url(st.session_state["random_img"]))
        st.write(st.session_state["random_img"])


def main() -> None:
    """
    Main method that sets up the streamlit app and builds the visualisation.
    """

    if "random_img" not in st.session_state:
        st.session_state["random_img"] = None

    st.session_state["collection"] = "./data/vectors/test.db"

    st.set_page_config(layout="wide", page_title="Phenocam image embeddings")

    st.title("Image embeddings")

    st.session_state["random_img"] = random_image()
    show_random_image()

    st.text("<-- random image")

    st.button("try again", on_click=random_image)

    # TODO figure out how streamlit is supposed to work
    closest_grid(st.session_state["random_img"])


if __name__ == "__main__":
    main()
