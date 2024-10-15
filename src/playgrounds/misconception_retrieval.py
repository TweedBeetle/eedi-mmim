import argparse

import weaviate
from loguru import logger
from typing import List, Dict
from tqdm import tqdm

from src.models import Misconception
from src.data_loader import load_misconception_mapping
from src.constants import mmim_data_path


def connect_weaviate():
    """
    Connect to the local embedded Weaviate instance.
    """
    client = weaviate.connect_to_embedded()

    if not client.is_ready():
        logger.error("Embedded Weaviate instance is not ready.")
        raise ConnectionError("Unable to connect to embedded Weaviate.")

    logger.info("Connected to embedded Weaviate successfully.")
    return client


def define_misconception_collection(client: weaviate.Client, collection_name: str = "Misconception"):
    """
    Define a new collection in embedded Weaviate for storing misconceptions.
    """
    schema = {
        "class": collection_name,
        "description": "A collection of mathematical misconceptions.",
        "properties": [
            {
                "name": "MisconceptionId",
                "dataType": ["int"],
                "description": "Unique identifier for the misconception."
            },
            {
                "name": "MisconceptionName",
                "dataType": ["text"],
                "description": "The name or description of the misconception."
            }
        ],
        "vectorizer": "none"  # We'll use custom vectors
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


import ollama


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
        # Generate embedding using Ollama
        embedding = ollama.embed(model='nomic-embed-text', input=mc.MisconceptionName)

        obj = {
            "MisconceptionId": mc.MisconceptionId,
            "MisconceptionName": mc.MisconceptionName,
            "vector": embedding
        }
        objects_to_insert.append(obj)

    logger.info("Inserting embeddings into Weaviate...")
    try:
        with client.batch as batch:
            for obj in objects_to_insert:
                batch.add_data_object(
                    data_object=obj,
                    class_name=collection_name
                )
        logger.info("Misconceptions inserted successfully.")
    except Exception as e:
        logger.exception("Failed to insert misconceptions into Weaviate.")
        raise

    return objects_to_insert


import ollama


def test_retrieval(client: weaviate.Client, query: str, collection_name: str = "Misconception", k: int = 5):
    """
    Perform a test retrieval of misconceptions based on the input query.
    """
    logger.info(f"Performing a test retrieval for query: '{query}'")
    try:
        # Generate embedding for the query
        query_embedding = ollama.embed(model='nomic-embed-text', input=query)

        response = client.query.get(collection_name, ["MisconceptionId", "MisconceptionName"]) \
            .with_near_vector({"vector": query_embedding, "distance": 0.7}) \
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
