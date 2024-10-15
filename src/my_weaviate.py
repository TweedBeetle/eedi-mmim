import argparse

import weaviate
import weaviate.classes as wvc
from loguru import logger
from typing import List, Dict
from tqdm import tqdm
from weaviate.client import WeaviateClient

import ollama
from weaviate.collections.classes.config import Configure
from weaviate.collections.classes.data import DataObject
from weaviate.embedded import EmbeddedOptions

from src.models import Misconception
from src.data_loader import load_misconceptions
from src.constants import mmim_data_path, PROJECT_ROOT


def get_weaviate_client():
    """
    Connect to the local embedded Weaviate instance.
    """
    client = weaviate.WeaviateClient(
        embedded_options=EmbeddedOptions(
            additional_env_vars={
                "ENABLE_MODULES": "backup-filesystem,generative-ollama,text2vec-ollama",
                "BACKUP_FILESYSTEM_PATH": str(PROJECT_ROOT / "out" / "weaviate_backups"),
            }
        )
        # Add additional options here (see Python client docs for syntax)
    )

    client.connect()

    if not client.is_ready():
        logger.error("Embedded Weaviate instance is not ready.")
        raise ConnectionError("Unable to connect to embedded Weaviate.")

    logger.info("Connected to embedded Weaviate successfully.")
    return client


def define_misconception_collection(client: WeaviateClient, collection_name: str = "Misconception", overwrite: bool = True):
    """
    Define a new collection in embedded Weaviate for storing misconceptions.
    If the collection already exists, either delete and recreate it or skip creation based on the overwrite parameter.

    Args:
        client: WeaviateClient instance
        collection_name: Name of the collection to create
        overwrite: If True, delete and recreate the collection if it exists. If False, skip creation if it exists.
    """
    try:
        # Check if the collection exists
        if client.collections.exists(collection_name):
            if overwrite:
                logger.warning(f"Collection '{collection_name}' already exists. Deleting and recreating.")
                client.collections.delete(collection_name)
            else:
                logger.info(f"Collection '{collection_name}' already exists. Skipping creation.")
                return

        misconceptions = client.collections.create(
            name=collection_name,
            description="A collection of mathematical misconceptions.",
            properties=[
                wvc.config.Property(
                    name="misconception_id",
                    data_type=wvc.config.DataType.INT,
                    description="Unique identifier for the misconception."
                ),
                wvc.config.Property(
                    name="misconception_name",
                    data_type=wvc.config.DataType.TEXT,
                    description="The name or description of the misconception."
                )
            ],
            vectorizer_config=[
                Configure.NamedVectors.text2vec_ollama(
                    name="MisconceptionNameVector",
                    source_properties=["misconception_name"],
                    api_endpoint="http://localhost:11434",
                    model="nomic-embed-text",
                )
            ],
        )
        logger.info(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        logger.exception(f"Failed to create collection '{collection_name}': {e}")
        raise


def embed_misconceptions(client: WeaviateClient, collection_name: str = "Misconception", limit: int = None) -> List[
    DataObject]:
    """
    Load misconceptions, generate embeddings using Ollama's nomic-embed-text, and insert into Weaviate.
    
    Args:
        client: WeaviateClient instance
        collection_name: Name of the collection to insert misconceptions into
        limit: Maximum number of misconceptions to ingest. If None, ingest all.
    """
    logger.info("Loading misconceptions from dataset...")
    misconceptions = load_misconceptions(str(mmim_data_path / "misconception_mapping.csv"))
    total_misconceptions = len(misconceptions)
    logger.info(f"Loaded {total_misconceptions} misconceptions.")

    if limit is not None:
        if limit < total_misconceptions:
            logger.warning(f"Limiting ingestion to {limit} misconceptions out of {total_misconceptions}.")
            misconceptions = misconceptions[:limit]
        else:
            logger.info(
                f"Limit {limit} is greater than or equal to total misconceptions. Ingesting all {total_misconceptions}."
            )

    # Prepare data for insertion
    objects_to_insert = []
    for mc in tqdm(misconceptions, desc="Embedding Misconceptions"):
        # Generate embedding using Ollama
        text_to_embed = f"search_document: {mc.misconception_name}"
        embedding_response = ollama.embed(model='nomic-embed-text', input=[text_to_embed])

        # Extract the actual embedding vector from the response
        embedding_vector = embedding_response['embeddings'][0]

        obj = wvc.data.DataObject(
            properties={
                "misconception_id": mc.misconception_id,
                "misconception_name": mc.misconception_name,
            },
            vector=embedding_vector
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


def retrieve_misconceptions(client: WeaviateClient, query: str, collection_name: str = "Misconception", k: int = 5):
    """
    Perform a test retrieval of misconceptions based on the input query.
    """
    logger.info(f"Performing a test retrieval for query: '{query}'")

    search_query_prefix = "search_query: "

    if not query.startswith(search_query_prefix):
        query = search_query_prefix + query

    try:
        collection = client.collections.get(collection_name)
        results = collection.query.near_text(
            query=query,
            limit=k,
            return_properties=["misconception_id", "misconception_name"]
        )

        if not results.objects:
            logger.warning("No misconceptions found for the given query.")
            return

        logger.info(f"Top {k} misconceptions related to '{query}':")
        for idx, obj in enumerate(results.objects, start=1):
            logger.info(
                f"{idx}. ID: {obj.properties['misconception_id']}, Name: {obj.properties['misconception_name']}"
            )
    except Exception as e:
        logger.exception("Failed to perform test retrieval.")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Embed misconceptions using Ollama's nomic-embed-text and perform a test retrieval."
    )
    parser.add_argument(
        "--query", type=str, default="Doesn't understand division", help="The query string to perform test retrieval."
    )
    parser.add_argument("--collection", type=str, default="Misconception", help="The name of the Weaviate collection.")
    parser.add_argument("--limit", type=int, default=5, help="Number of top misconceptions to retrieve.")
    parser.add_argument(
        # "--ingest_limit", type=int, default=None,
        "--ingest_limit", type=int, default=100,
        help="Maximum number of misconceptions to ingest. If not specified, ingest all."
    )
    args = parser.parse_args()

    pipeline(args)


def pipeline(args):
    try:
        client = get_weaviate_client()
        define_misconception_collection(client, args.collection)
        embed_misconceptions(client, args.collection, args.ingest_limit)
        retrieve_misconceptions(client, args.query, args.collection, args.limit)
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
    # pipeline()
