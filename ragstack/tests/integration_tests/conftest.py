import os
import logging
from typing import Callable
import pytest
from pathlib import Path

from langchain_core.embeddings import Embeddings


def pytest_configure():
    data_path = Path(__file__).parent.absolute() / "data"

    pytest.EMBEDDING_PATH = data_path / "embedding.json"

    # validate that all the paths are correct and the files exist
    for path in [
        pytest.EMBEDDING_PATH,
    ]:
        assert (
            path.exists()
        ), f"File {path} does not exist. Available files: {list(data_path.iterdir())}"


LOGGER = logging.getLogger(__name__)
DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def _load_env() -> None:
    dotenv_path = os.path.join(DIR_PATH, os.pardir, ".env")
    if os.path.exists(dotenv_path):
        from dotenv import load_dotenv

        load_dotenv(dotenv_path)


_load_env()


def get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        LOGGER.warning(f"Missing environment variable: {name}")
        pytest.skip(f"Missing environment variable: {name}")

    return value


class MockEmbeddings(Embeddings):
    def __init__(self):
        self.embedded_documents = None
        self.embedded_query = None

    @staticmethod
    def mock_embedding(text: str):
        return [len(text) / 2, len(text) / 5, len(text) / 10]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_documents = texts
        return [self.mock_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded_query = text
        return self.mock_embedding(text)


@pytest.fixture
def embedding_flow() -> str:
    with open(pytest.EMBEDDING_PATH, "r") as f:
        return f.read()


@pytest.fixture
def astradb_component() -> Callable:
    from langflow.components.vectorstores import AstraDBVectorStoreComponent

    def component_builder(
        embedding: Embeddings = MockEmbeddings,
        collection: str = "test",
        inputs: list = [],
    ):
        token = get_env_var("ASTRA_DB_APPLICATION_TOKEN")
        api_endpoint = get_env_var("ASTRA_DB_API_ENDPOINT")
        return AstraDBVectorStoreComponent().build(
            embedding=embedding,
            collection_name=collection,
            inputs=inputs,
            token=token,
            api_endpoint=api_endpoint,
        )

    return component_builder
