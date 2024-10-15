import argparse

import weaviate
import os
from loguru import logger
from typing import List, Dict
from tqdm import tqdm

from src.models import Misconception
from src.data_loader import load_misconception_mapping
from src.constants import mmim_data_path


def connect_weaviate():
    """
    Connect to the Weaviate instance using environment variables.
    Ensure that the following environment variables are set:
    - WEAVIATE_URL: The URL of your Weaviate instance
    - WEAVIATE_API_KEY: Your Weaviate API key
    - OPENAI_APIKEY: Your OpenAI API key for the Ollama vectorizer
    """
    WEAVIATE_URL = os.getenv("WEAVIATE_URL")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
    OPENAI_APIKEY = os.getenv("OPENAI_APIKEY")

    if not WEAVIATE_URL or not WEAVIATE_API_KEY or not OPENAI_APIKEY:
        logger.error("One or more required environment variables are missing.")
        raise EnvironmentError("Please set WEAVIATE_URL, WEAVIATE_API_KEY, and OPENAI_APIKEY.")

    client = weaviate.Client(
        url=WEAVIATE_URL,
        auth_client_secret=weaviate.auth.AuthApiKey(api_key=WEAVIATE_API_KEY),
        additional_headers={
            "X-OpenAI-Api-Key": OPENAI_APIKEY
        }
    )

    if not client.is_ready():
        logger.error("Weaviate instance is not ready. Please check the connection details.")
        raise ConnectionError("Unable to connect to Weaviate.")

    logger.info("Connected to Weaviate successfully.")
    return client


def define_misconception_collection(client: weaviate.Client, collection_name: str = "Misconception"):
    """
    Define a new collection in Weaviate for storing misconceptions with the Ollama vectorizer.
    """
    schema = {
        "class": collection_name,
        "description": "A collection of mathematical misconceptions.",
        "vectorizer": "text2vec-ollama",
        "moduleConfig": {
            "text2vec-ollama": {
                "model": "nomic-embed-text",
                "api_endpoint": "127.0.0.1:11434",
            }
        },
        "properties": [
            {
                "name": "MisconceptionId",
                "dataType": ["int"],
                "description": "Unique identifier for the misconception.",
                "moduleConfig": {
                    "text2vec-ollama": {
                        "vectorizePropertyName": False  # Do not prepend property name to the value
                    }
                }
            },
            {
                "name": "MisconceptionName",
                "dataType": ["text"],
                "description": "The name or description of the misconception.",
                "moduleConfig": {
                    "text2vec-ollama": {
                        "vectorizePropertyName": False  # Do not prepend property name to the value
                    }
                }
            }
        ],
        "replicationConfig": {
            "factor": 1
        },
        "shardingConfig": {
            "virtualPerPhysical": 128,
            "desiredCount": 1,
            "actualCount": 1,
            "desiredVirtualCount": 128,
            "actualVirtualCount": 128,
            "key": "_id",
            "strategy": "hash",
            "function": "murmur3"
        },
        "vectorIndexConfig": {
            "distance": "cosine",
            "efConstruction": 128,
            "ef": 128,
            "dynamicEfMin": 100,
            "dynamicEfMax": 500,
            "dynamicEfFactor": 8,
            "vectorCacheMaxObjects": 1000000,
            "flatSearchCutoff": 40000,
            "maxConnections": 32
        },
        "vectorIndexType": "hnsw"
    }

    try:
        if client.schema.exists(collection_name):
            logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        else:
            client.schema.create_class(schema)
            logger.info(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        logger.exception(f"Failed to create collection '{collection_name}': {e}")
        raise


def embed_misconceptions(client: weaviate.Client, collection_name: str = "Misconception") -> List[Dict]:
    """
    Load all misconceptions, generate embeddings using Ollama's nomic-embed-text, and insert into Weaviate.
    """
    logger.info("Loading misconceptions from dataset...")
    misconceptions = load_misconception_mapping(str(mmim_data_path / "misconception_mapping.csv"))
    logger.info(f"Loaded {len(misconceptions)} misconceptions.")

    # Prepare data for insertion
    objects_to_insert = []
    for mc in tqdm(misconceptions, desc="Embedding Misconceptions"):
        obj = {
            "MisconceptionId": mc.MisconceptionId,
            "MisconceptionName": mc.MisconceptionName
        }
        objects_to_insert.append(obj)

    logger.info("Inserting embeddings into Weaviate...")
    try:
        client.batch.add_objects(
            objects_to_insert,
            class_name=collection_name
        )
        logger.info("Misconceptions inserted successfully.")
    except Exception as e:
        logger.exception("Failed to insert misconceptions into Weaviate.")
        raise

    return objects_to_insert


def test_retrieval(client: weaviate.Client, query: str, collection_name: str = "Misconception", k: int = 5):
    """
    Perform a test retrieval of misconceptions based on the input query.
    """
    logger.info(f"Performing a test retrieval for query: '{query}'")
    try:
        response = client.query.get(collection_name, ["MisconceptionId", "MisconceptionName"]) \
            .with_near_text({"concepts": [query], "distance": 0.7}) \
            .with_limit(k) \
            .do()

        results = response.get("data", {}).get("Get", {}).get(collection_name, [])
        if not results:
            logger.warning("No misconceptions found for the given query.")
            return

        logger.info(f"Top {k} misconceptions related to '{query}':")
        for idx, res in enumerate(results, start=1):
            logger.info(f"{idx}. ID: {res['MisconceptionId']}, Name: {res['MisconceptionName']}")
    except Exception as e:
        logger.exception("Failed to perform test retrieval.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Embed all misconceptions using Ollama's nomic-embed-text and perform a test retrieval."
        )
    parser.add_argument("--query", type=str, default="biology", help="The query string to perform test retrieval.")
    parser.add_argument("--collection", type=str, default="Misconception", help="The name of the Weaviate collection.")
    parser.add_argument("--limit", type=int, default=5, help="Number of top misconceptions to retrieve.")
    args = parser.parse_args()

    try:
        client = connect_weaviate()
        define_misconception_collection(client, args.collection)
        embed_misconceptions(client, args.collection)
        test_retrieval(client, args.query, args.collection, args.limit)
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
