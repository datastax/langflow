import os
import orjson
from typing import Callable

from langchain_core.documents import Document
from langflow.load import run_flow_from_json
from langflow.schema import Record

BASIC_COLLECTION = "test"
EMBEDDING_FLOW_COLLECTION = "test_embedding_flow"


def test_build_no_inputs(astradb_component: Callable):
    astradb_component()


def test_build_with_inputs(astradb_component: Callable):
    record = Record.from_document(Document(page_content="test"))
    record2 = Record.from_document(Document(page_content="test2"))
    inputs = [record, record2]
    astradb_component(collection=BASIC_COLLECTION, inputs=inputs)


def test_astra_embedding_flow(embedding_flow: str):
    """
    Embeds the contents of a URL into AstraDB.
    """
    flow = orjson.loads(embedding_flow)
    TWEAKS = {
        "AstraDB-s9tdG": {
            "token": os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            "api_endpoint": os.environ["ASTRA_DB_API_ENDPOINT"],
            "collection_name": EMBEDDING_FLOW_COLLECTION,
        },
        "SplitText-v9ZHX": {},
        "URL-vWSxt": {},
        "OpenAIEmbeddings-YQwtD": {"openai_api_key": os.environ["OPENAI_API_KEY"]},
    }

    result = run_flow_from_json(flow=flow, input_value="message", tweaks=TWEAKS)
    # Shouldn't have anything really specific to verify here, just checking that it runs
    assert result is not None
