import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument("--query", help="Query")
parser.add_argument("--book", help="Book")

args=parser.parse_args()

if args.query is None:
    print("Please provide a query")
    sys.exit(1)

if args.book is None:
    print("Please provide a book")
    sys.exit(1)

import pandas as pd
import openai
import numpy as np

from dotenv import load_dotenv
import os

# Use load_env to trace the path of .env:
load_dotenv('.env')
openai.api_key = os.environ["OPENAI_API_KEY"]

QUERY_EMBEDDINGS_MODEL = "text-embedding-ada-002"

def get_query_embedding(text: str) -> list[float]:
    return get_embedding(text, QUERY_EMBEDDINGS_MODEL)

def get_embedding(text: str, model: str) -> list[float]:
    result = openai.Embedding.create(
      model=model,
      input=text
    )
    return result["data"][0]["embedding"]

def order_document_sections_by_query_similarity(query: str, contexts: dict[(str, str), np.array]) -> list[(float, (str, str))]:
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections.
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_query_embedding(query)

    document_similarities = sorted([
        (vector_similarity(query_embedding, doc_embedding), doc_index) for doc_index, doc_embedding in contexts.items()
    ], reverse=True)

    return document_similarities

def vector_similarity(x: list[float], y: list[float]) -> float:
    """
    We could use cosine similarity or dot product to calculate the similarity between vectors.
    In practice, we have found it makes little difference.
    """
    # return np.dot(np.array(x), np.array(y))
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

def load_embeddings(fname: str) -> dict[tuple[str, str], list[float]]:
    """
    Read the document embeddings and their keys from a CSV.
    fname is the path to a CSV with exactly these named columns:
        "title", "0", "1", ... up to the length of the embedding vectors.
    """

    df = pd.read_csv(fname, header=0)
    max_dim = max([int(c) for c in df.columns if c != "title"])
    return {
           (r.title): [r[str(i)] for i in range(max_dim + 1)] for _, r in df.iterrows()
    }

document_embeddings = load_embeddings(f'{args.book}.embeddings.csv')

document_similarties = order_document_sections_by_query_similarity(
  args.query,
  document_embeddings
)

for (sim, page) in document_similarties:
  print(f"{page}      {sim}")
