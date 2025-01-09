import logging
import sqlite3
import struct
from abc import ABCMeta, abstractmethod
from datetime import datetime
from typing import List, Optional

import sqlite_vec

from phenocam.data.db_config import SQLITE_SCHEMA

logging.basicConfig(level=logging.INFO)


def serialize_f32(vector: List[float]) -> bytes:
    """
    Serialize a list of floats into a compact "raw bytes" format.

    :param vector: List of floats to serialize.
    :type vector: List[float]
    :return: Serialized bytes.
    :rtype: bytes
    """
    return struct.pack("%sf" % len(vector), *vector)


def deserialize(packed: bytes) -> List[float]:
    """
    Deserialize bytes back into a list of floats.

    :param packed: Serialized bytes.
    :type packed: bytes
    :return: List of floats.
    :rtype: List[float]
    """
    size = int(len(packed) / 4)
    return struct.unpack("%sf" % size, packed)


class VectorStore(metaclass=ABCMeta):
    @abstractmethod
    def add(self, url: str, embeddings: List[float]) -> None:
        """
        Add a vector to the store.

        :param url: URL associated with the vector.
        :type url: str
        :param embeddings: List of float embeddings.
        :type embeddings: List[float]
        """
        pass

    @abstractmethod
    def get(self, url: str) -> List[float]:
        """
        Retrieve a vector from the store.

        :param url: URL associated with the vector.
        :type url: str
        :return: List of float embeddings.
        :rtype: List[float]
        """
        pass

    @abstractmethod
    def closest(self, embeddings: List[float]) -> List[float]:
        """
        Find the closest vectors to the given embeddings.

        :param embeddings: List of float embeddings.
        :type embeddings: List[float]
        :return: List of closest float embeddings.
        :rtype: List[float]
        """
        pass

    @abstractmethod
    def embeddings(self) -> List[List[float]]:
        """
        Retrieve all embeddings from the store.

        :return: List of all embeddings.
        :rtype: List[List[float]]
        """
        pass

    @abstractmethod
    def ids(self) -> List[str]:
        """
        Retrieve all IDs from the store.

        :return: List of all IDs.
        :rtype: List[str]
        """
        pass


class PostgresStore(VectorStore):
    def __init__(self, db_name: str):
        """
        Initialize a PostgresStore instance.

        :param db_name: Name of the PostgreSQL database.
        :type db_name: str
        """
        self.db_name = db_name

    def add(self, url: str, embeddings: List[float]) -> None:
        """
        Add a vector to the Postgres store.

        :param url: URL associated with the vector.
        :type url: str
        :param embeddings: List of float embeddings.
        :type embeddings: List[float]
        """
        pass

    def get(self, url: str) -> List[float]:
        """
        Retrieve a vector from the Postgres store.

        :param url: URL associated with the vector.
        :type url: str
        :return: List of float embeddings.
        :rtype: List[float]
        """
        pass

    def closest(self, embeddings: List[float], n_results: int = 25) -> List[str]:
        """
        Find the closest vectors to the given embeddings in the Postgres store.

        :param embeddings: List of float embeddings.
        :type embeddings: List[float]
        :param n_results: Number of closest results to return, defaults to 25.
        :type n_results: int
        :return: List of closest URLs.
        :rtype: List[str]
        """
        pass

    def embeddings(self) -> List[List[float]]:
        """
        Retrieve all embeddings from the Postgres store.

        :return: List of all embeddings.
        :rtype: List[List[float]]
        """
        pass

    def ids(self) -> List[str]:
        """
        Retrieve all IDs from the Postgres store.

        :return: List of all IDs.
        :rtype: List[str]
        """
        pass


class SQLiteVecStore(VectorStore):
    def __init__(self, db_name: str, embedding_len: Optional[int] = 2048, check_same_thread: bool = True):
        """
        Initialize a SQLiteVecStore instance.

        :param db_name: Name of the SQLite database.
        :type db_name: str
        :param embedding_len: Length of the embeddings, defaults to 2048.
        :type embedding_len: Optional[int]
        :param check_same_thread: Whether to check the same thread, defaults to True.
        :type check_same_thread: bool
        """
        self._check_same_thread = check_same_thread
        self.embedding_len = embedding_len
        self.load_ext(db_name)
        self.load_schema()

    def load_ext(self, db_name: str) -> None:
        """
        Load the SQLite extension into the database if needed.

        :param db_name: Name of the SQLite database.
        :type db_name: str
        """
        db = sqlite3.connect(db_name, check_same_thread=self._check_same_thread)
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        self.db = db

    def load_schema(self) -> None:
        """
        Load the database schema if needed.
        Default embedding length is 2048, set at init
        """
        query = SQLITE_SCHEMA.format(self.embedding_len)

        try:
            self.db.execute(query)
        except sqlite3.OperationalError as err:
            if "already exists" in str(err):
                pass
            else:
                raise

    def add(
        self,
        url: str,
        embeddings: List[float],
        classification: Optional[str] = "",
        date: Optional[datetime] = None,
        site: Optional[str] = "",
    ) -> None:
        """
        Add a vector to the SQLite-vec store.

        :param url: URL associated with the vector.
        :type url: str
        :param embeddings: List of float embeddings.
        :type embeddings: List[float]
        :param classification: Classification label, defaults to "".
        :type classification: Optional[str]
        :param date: Date associated with the vector, defaults to None.
        :type date: Optional[datetime]
        :param site: Site associated with the vector, defaults to "".
        :type site: Optional[str]
        """
        self.db.execute(
            "INSERT INTO embeddings(url, embedding, classification, date, site) VALUES (?, ?, ?, ?, ?)",
            [url, serialize_f32(embeddings), classification, date, site],
        )
        self.db.commit()

    def get(self, url: str) -> List[float]:
        """
        Retrieve a vector from the SQLite-vec store.

        :param url: URL associated with the vector.
        :type url: str
        :return: List of float embeddings.
        :rtype: List[float]
        """
        result = self.db.execute("select embedding from embeddings where url = ?", [url]).fetchone()
        if result and len(result):
            return result[0]
        else:
            return None

    def closest(self, embeddings: List[float], n_results: int = 25) -> List[str]:
        """
        Find and return the N closest examples by distance.

        :param embeddings: List of float embeddings.
        :type embeddings: List[float]
        :param n_results: Number of closest results to return, defaults to 25.
        :type n_results: int
        :return: List of closest URLs.
        :rtype: List[str]
        """
        # https://github.com/asg017/sqlite-vec/issues/41 - "limit ?" not guaranteed
        # Note - stopped returning distance for consistency, but might be useful
        query = """select
            url
            from embeddings
            where embedding match ?
                and k = ?
            order by distance;
        """
        results = self.db.execute(query, [embeddings, n_results]).fetchall()
        return [i for j in results for i in j]

    def labelled(self, label: str, n_results: int = 50) -> List[str]:
        """
        Retrieve URLs of vectors with the specified classification label.

        :param label: Classification label to filter by.
        :type label: str
        :param n_results: Number of results to return, defaults to 50.
        :type n_results: int
        :return: List of URLs with the specified classification label.
        :rtype: List[str]
        """
        labelled = self.db.execute(
            """select url from embeddings where classification = ? limit ?""", (label, n_results)
        ).fetchall()
        return [i for j in labelled for i in j]

    def random(self) -> str:
        """Return the URL of a random image from the store."""
        random = self.db.execute("""select url from embeddings order by random() limit 1""").fetchone()
        if random and len(random):
            return random[0]

    def classes(self) -> List[str]:
        """
        Retrieve all distinct classification labels from the store.

        :return: List of distinct classification labels.
        :rtype: List[str]
        """
        classes = self.db.execute("""select distinct classification from embeddings""").fetchall()
        return [i for j in classes for i in j]

    def embeddings(self) -> List[List[float]]:
        """
        Retrieve all embeddings from the SQLite-vec store.

        :return: List of all embeddings.
        :rtype: List[List[float]]
        """
        embeddings = self.db.execute("""select embedding from embeddings""").fetchall()
        return [deserialize(i) for j in embeddings for i in j]

    def ids(self) -> List[str]:
        """
        Retrieve all URLs from the SQLite-vec store.

        :return: List of all URLs.
        :rtype: List[str]
        """
        urls = self.db.execute("""select url from embeddings""").fetchall()
        return [i for j in urls for i in j]


def vector_store(
    store_type: Optional[str] = "sqlite", db_name: Optional[str] = "test_collection", **kwargs
) -> VectorStore:
    """
    Factory function to create a vector store instance.

    :param store_type: Type of the store, either "sqlite" or "postgres", defaults to "sqlite".
    :type store_type: Optional[str]
    :param db_name: Name of the database, defaults to "test_collection".
    :type db_name: Optional[str]
    :return: Vector store instance.
    :rtype: VectorStore
    :raises ValueError: If an unknown store type is provided.
    """
    if store_type == "postgres":
        return PostgresStore(db_name, **kwargs)
    elif store_type == "sqlite":
        return SQLiteVecStore(db_name, **kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
