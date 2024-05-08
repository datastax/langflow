import os
import orjson
from typing import Callable

from langchain_core.documents import Document
from langflow.load import run_flow_from_json


def test_build_no_inputs(astradb_component: Callable):
    astradb_component()


def test_build_with_inputs(astradb_component: Callable):
    doc = Document("test")
    inputs = [doc]
    astradb_component(inputs=inputs)


def test_astra_flow(embedding_flow: str):
    flow = orjson.loads(embedding_flow)
    TWEAKS = {
        "AstraDB-s9tdG": {
            "token": os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            "api_endpoint": os.environ["ASTRA_DB_API_ENDPOINT"],
            "collection_name": "test",
        },
        "SplitText-v9ZHX": {},
        "URL-vWSxt": {},
        "OpenAIEmbeddings-YQwtD": {"openai_api_key": os.environ["OPENAI_API_KEY"]},
    }

    result = run_flow_from_json(flow=flow, input_value="message", tweaks=TWEAKS)
    print(result)
