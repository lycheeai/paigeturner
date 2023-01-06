import argparse, sys

parser=argparse.ArgumentParser()

parser.add_argument("--pdf", help="Name of PDF")

args=parser.parse_args()

if args.pdf is None:
    print("Please provide a PDF filename")
    sys.exit(1)

filename = f"{args.pdf}"

from dotenv import load_dotenv
import os

# Use load_env to trace the path of .env:
load_dotenv('.env')

import pandas as pd
from typing import Set
from transformers import GPT2TokenizerFast

import numpy as np

from PyPDF2 import PdfReader

import pandas as pd
import openai
import csv
import numpy as np
import os
import pickle
import time
from transformers import GPT2TokenizerFast

openai.api_key = os.environ["OPENAI_API_KEY"]

DOC_EMBEDDINGS_MODEL = f"text-embedding-ada-002"
EMBEDDING_SIZE=1536

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def count_tokens(text: str) -> int:
  """count the number of tokens in a string"""
  return len(tokenizer.encode(text))

def extract_pages(
    page_text: str,
    index: int,
) -> str:
  """
  Extract the text from the page
  """
  if len(page_text) == 0:
    return []

  content = " ".join(page_text.split())
  # print("page text: " + content)
  outputs = [("Page " + str(index), content, count_tokens(content)+4)]

  return outputs

reader = PdfReader(filename)

res = []
i = 1
for page in reader.pages:
  res += extract_pages(page.extract_text(), i)
  i += 1
df = pd.DataFrame(res, columns=["title", "content", "tokens"])
df = df[df.tokens<2046]
df = df.reset_index().drop('index',axis=1) # reset index
df.head()

df.to_csv(f'{filename}.pages.csv', index=False)

def get_embedding(text: str, model: str) -> list[float]:
  result = openai.Embedding.create(
    model=model,
    input=text
  )
  return result["data"][0]["embedding"]

def get_doc_embedding(text: str) -> list[float]:
  res = get_embedding(text, DOC_EMBEDDINGS_MODEL)
  print(f"Embedding: {len(text)} {len(res)}")
  return res

def compute_doc_embeddings(df: pd.DataFrame) -> dict[tuple[str], list[float]]:
  """
  Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
  Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
  """
  return {
    idx: get_doc_embedding(r.content) for idx, r in df.iterrows()
  }

# CSV with exactly these named columns:
# "title", "0", "1", ... up to the length of the embedding vectors.

doc_embeddings = compute_doc_embeddings(df)

with open(f'{filename}.embeddings.csv', 'w') as f:
  writer = csv.writer(f)
  writer.writerow(["title"] + list(range(EMBEDDING_SIZE)))
  for i, embedding in list(doc_embeddings.items()):
    writer.writerow(["Page " + str(i + 1)] + embedding)
