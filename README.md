# Earnings Call Q&A with RAG
## AI-powered question answering system for financial earnings call transcripts.

Stock traders and investors faces a problem: quarterly earnings call transcripts are 50,000+ words of dense financial discussions between company executives and analysts. Reading and analyzing multiple transcripts takes hours of valuable time.
THis project solves that problem: get specific answers from earning calls in seconds not hours.
Ask questions like "What were the revenue drivers?" or "What risks did management mention?" and get accurate answers instantly.

## Tech Stack

- **ChromaDB** - Vector database for semantic search
- **OpenAI API** - Embeddings (text-embedding-3-small) and LLM (gpt-4o-mini)
- **LangChain** - Text splitting and chunking utilities
- **Python 3.10+**

## How It Works

1. **Load Transcripts** - Reads earnings call transcripts from local files.

2. **Chunk Documents** - Splits transcripts into 1000-character chunks with 200-character overlap for optimal retrieval

3. **Generate Embeddings** - Creates vector embeddings using OpenAI's text-embedding-3-small model

4. **Store in Vector Database** - Saves embeddings and text chunks in ChromaDB with persistent storage

5. **Query Processing** - User asks a question → system embeds the query using the same model

6. **Semantic Search** - Retrieves top-k most relevant chunks using vector similarity

7. **Answer Generation** - Sends query + relevant chunks to GPT-4o-mini, which generates a focused answer with specific metrics and numbers
