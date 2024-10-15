import ollama
from src.data_loader import load_misconception_mapping
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def embed_misconceptions(model_name: str = 'nomic-embed-text') -> Dict[int, List[float]]:
    """
    Embeds all misconceptions using the specified Ollama model.

    Args:
        model_name (str): The name of the Ollama embedding model to use.

    Returns:
        Dict[int, List[float]]: A dictionary mapping MisconceptionId to its embedding vector.
    """
    misconceptions = load_misconception_mapping(str(mmim_data_path / "misconception_mapping.csv"))
    embeddings = {}
    
    logger.info("Embedding misconceptions using Ollama...")
    
    for mc in misconceptions:
        embedding_response = ollama.embed(model=model_name, input=mc.MisconceptionName)
        # Assuming the response contains the embedding under the key 'embedding'
        embedding = embedding_response.get('embedding')
        if embedding:
            embeddings[mc.MisconceptionId] = embedding
        else:
            logger.warning(f"Embedding not found for MisconceptionId {mc.MisconceptionId}")
    
    logger.info(f"Successfully embedded {len(embeddings)} misconceptions.")
    return embeddings

def test_retrieval(embeddings: Dict[int, List[float]], query: str, top_n: int = 5) -> List[Dict]:
    """
    Performs a test retrieval by embedding the query and finding top N similar misconceptions.

    Args:
        embeddings (Dict[int, List[float]]): A dictionary of MisconceptionId to embedding vectors.
        query (str): The query string to retrieve similar misconceptions.
        top_n (int): Number of top similar misconceptions to retrieve.

    Returns:
        List[Dict]: A list of dictionaries containing MisconceptionId, MisconceptionName, and similarity score.
    """
    logger.info("Embedding the query for retrieval...")
    query_embedding_response = ollama.embed(model='nomic-embed-text', input=query)
    query_embedding = query_embedding_response.get('embedding')
    
    if not query_embedding:
        logger.error("Failed to obtain embedding for the query.")
        return []
    
    logger.info("Calculating cosine similarities...")
    similarity_scores = {}
    for mc_id, mc_embedding in embeddings.items():
        similarity = cosine_similarity([query_embedding], [mc_embedding])[0][0]
        similarity_scores[mc_id] = similarity
    
    # Sort misconceptions by similarity score in descending order
    sorted_misconceptions = sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)
    
    top_misconceptions = []
    misconceptions = load_misconception_mapping(str(mmim_data_path / "misconception_mapping.csv"))
    mc_lookup = {mc.MisconceptionId: mc.MisconceptionName for mc in misconceptions}
    
    for mc_id, score in sorted_misconceptions[:top_n]:
        top_misconceptions.append({
            "MisconceptionId": mc_id,
            "MisconceptionName": mc_lookup.get(mc_id, "Unknown Misconception"),
            "SimilarityScore": score
        })
    
    logger.info(f"Top {top_n} similar misconceptions retrieved.")
    return top_misconceptions

if __name__ == "__main__":
    # Embed all misconceptions
    misconception_embeddings = embed_misconceptions(model_name='nomic-embed-text')
    
    # Define a sample query for test retrieval
    sample_query = "Why do objects fall towards the earth?"
    
    # Perform test retrieval
    top_misconceptions = test_retrieval(embeddings=misconception_embeddings, query=sample_query, top_n=5)
    
    # Display the results
    print(f"Top 5 misconceptions similar to the query '{sample_query}':")
    for mc in top_misconceptions:
        print(f"Misconception ID: {mc['MisconceptionId']}, Name: {mc['MisconceptionName']}, Similarity: {mc['SimilarityScore']:.4f}")
