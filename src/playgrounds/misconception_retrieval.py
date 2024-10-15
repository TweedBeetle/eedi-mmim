import argparse

import weaviate
import weaviate.classes as wvc
from loguru import logger
from typing import List, Dict
from tqdm import tqdm
from weaviate.client import WeaviateClient

import ollama

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


def define_misconception_collection(client: WeaviateClient, collection_name: str = "Misconception"):
    """
    Define a new collection in embedded Weaviate for storing misconceptions.
    """
    try:
        misconceptions = client.collections.create(
            name=collection_name,
            description="A collection of mathematical misconceptions.",
            properties=[
                wvc.config.Property(
                    name="MisconceptionId",
                    data_type=wvc.config.DataType.INT,
                    description="Unique identifier for the misconception."
                ),
                wvc.config.Property(
                    name="MisconceptionName",
                    data_type=wvc.config.DataType.TEXT,
                    description="The name or description of the misconception."
                )
            ],
            vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        if e.status_code == 422:
            logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
        else:
            logger.exception(f"Failed to create collection '{collection_name}': {e}")
            raise
    except Exception as e:
        logger.exception(f"Failed to create collection '{collection_name}': {e}")
        raise


def embed_misconceptions(client: WeaviateClient, collection_name: str = "Misconception", limit: int = None) -> List[Dict]:
    """
    Load misconceptions, generate embeddings using Ollama's nomic-embed-text, and insert into Weaviate.
    
    Args:
        client: WeaviateClient instance
        collection_name: Name of the collection to insert misconceptions into
        limit: Maximum number of misconceptions to ingest. If None, ingest all.
    """
    logger.info("Loading misconceptions from dataset...")
    misconceptions = load_misconception_mapping(str(mmim_data_path / "misconception_mapping.csv"))
    total_misconceptions = len(misconceptions)
    logger.info(f"Loaded {total_misconceptions} misconceptions.")

    if limit is not None:
        if limit < total_misconceptions:
            logger.warning(f"Limiting ingestion to {limit} misconceptions out of {total_misconceptions}.")
            misconceptions = misconceptions[:limit]
        else:
            logger.info(f"Limit {limit} is greater than or equal to total misconceptions. Ingesting all {total_misconceptions}.")

    # Prepare data for insertion
    objects_to_insert = []
    for mc in tqdm(misconceptions, desc="Embedding Misconceptions"):
        # Generate embedding using Ollama
        embedding = ollama.embed(model='nomic-embed-text', input=mc.MisconceptionName)

        obj = wvc.data.DataObject(
            properties={
                "MisconceptionId": mc.MisconceptionId,
                "MisconceptionName": mc.MisconceptionName,
            },
            vector=embedding
        )
        objects_to_insert.append(obj)

    logger.info("Inserting embeddings into Weaviate...")
    try:
        collection = client.collections.get(collection_name)
        collection.data.insert_many(objects_to_insert)
        logger.info(f"Successfully inserted {len(objects_to_insert)} misconceptions.")
    except Exception as e:
        logger.exception("Failed to insert misconceptions into Weaviate.")
        raise

    return objects_to_insert


def test_retrieval(client: WeaviateClient, query: str, collection_name: str = "Misconception", k: int = 5):
    """
    Perform a test retrieval of misconceptions based on the input query.
    """
    logger.info(f"Performing a test retrieval for query: '{query}'")
    try:
        # Generate embedding for the query
        query_embedding = ollama.embed(model='nomic-embed-text', input=query)

        collection = client.collections.get(collection_name)
        results = collection.query.near_vector(
            vector=query_embedding,
            limit=k,
            return_properties=["MisconceptionId", "MisconceptionName"]
        )

        if not results.objects:
            logger.warning("No misconceptions found for the given query.")
            return

        logger.info(f"Top {k} misconceptions related to '{query}':")
        for idx, obj in enumerate(results.objects, start=1):
            logger.info(f"{idx}. ID: {obj.properties['MisconceptionId']}, Name: {obj.properties['MisconceptionName']}")
    except Exception as e:
        logger.exception("Failed to perform test retrieval.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Embed misconceptions using Ollama's nomic-embed-text and perform a test retrieval."
    )
    parser.add_argument("--query", type=str, default="biology", help="The query string to perform test retrieval.")
    parser.add_argument("--collection", type=str, default="Misconception", help="The name of the Weaviate collection.")
    parser.add_argument("--limit", type=int, default=5, help="Number of top misconceptions to retrieve.")
    parser.add_argument("--ingest_limit", type=int, default=None, help="Maximum number of misconceptions to ingest. If not specified, ingest all.")
    args = parser.parse_args()

    try:
        client = connect_weaviate()
        define_misconception_collection(client, args.collection)
        embed_misconceptions(client, args.collection, args.ingest_limit)
        test_retrieval(client, args.query, args.collection, args.limit)
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
