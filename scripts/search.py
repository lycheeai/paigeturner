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

import openai

from dotenv import load_dotenv
import os

# Use load_env to trace the path of .env:
load_dotenv('.env')
openai.api_key = os.environ["OPENAI_API_KEY"]

from lib import load_embeddings, order_document_sections_by_query_similarity

document_embeddings = load_embeddings(f'{args.book}.embeddings.csv')

document_similarties = order_document_sections_by_query_similarity(
  args.query,
  document_embeddings
)

for (sim, page) in document_similarties[0:20]:
  print(f"{page}      {sim}")
