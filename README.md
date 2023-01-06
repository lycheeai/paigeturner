
## Setup

```bash
python3 -m venv venv

source venv/bin/activate

pip3 install -r requirements.txt
```


Add to .env file:
```
OPENAI_API_KEY=....
```

# Generating embeddings

```bash
python3 scripts/pdf_to_embeddings.py \
  --pdf ~/Desktop/Real_World_Cryptography.pdf
```
This will generate a *.embeddings.csv file and a *.pages.csv file

Move these files to a directory you can access.

# Querying

Get top pages related to this question
```bash
# Find related pages
python3 scripts/search.py \
	--query "What is the best way to secure passwords?" \
	--book examples/crypto.pdf
```

Ask a question and get an answer
```bash
python3 scripts/ask.py \
	--query "What is the best way to secure passwords?" \
	--book examples/crypto.pdf
```

Ask this question without using the book at all
```bash
python3 scripts/ask.py \
	--query "What is the best way to secure passwords?" \
  --basic \
	--book examples/crypto.pdf
```
