from typing import List
from chromadb import Collection, Client, PersistentClient
from openai import OpenAI
import os
import fileinput as f
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Initialize ChromaDB client and collection
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="earning_transcripts")

# Phase 1: Fetch earning transcripts
def fetch_earning_transcript() -> List[str]:
    """
    Fetches earning transcripts from the transcripts folder.

    Args:
        None

    Returns:
        List[str]: The fetched earning transcripts.
    """
    transcripts = []
    transcripts_folder = 'transcripts'
    
    for filename in os.listdir(transcripts_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(transcripts_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                transcripts.append(content)

    return transcripts

# Phase 2: Chunking transcripts
def chunk_transcripts(transcripts: List[str]) -> List[str]:
    """
    Chunks the earning transcripts into smaller pieces.

    Args:
        transcripts (List[str]): The earning transcripts to chunk.

    Returns:
        List[str]: The chunked earning transcripts.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_transcripts = []
    for transcript in transcripts:
        chunks = text_splitter.split_text(transcript)
        chunked_transcripts.extend(chunks)
    
    return chunked_transcripts

# Phase 3: Vectorize and store in ChromaDB
def vectorize_transcripts(chunked_transcripts: List[str], collection: Collection, openai_client: OpenAI):
    """
    Vectorizes the earning transcripts and stores them in the provided ChromaDB collection.

    Args:
        transcripts (List[str]): The earning transcripts to vectorize.
        collection (Collection): The ChromaDB collection to store the vectors.
        openai_client (OpenAI): The OpenAI client for embedding generation. 
    Returns:
        None
    """
    for i, transcript in enumerate(chunked_transcripts):
        embedding = openai_client.embeddings.create(
            input=transcript,
            model="text-embedding-3-small"
        )
        embedding = embedding.data[0].embedding
        
        collection.add(
            documents=[transcript],
            embeddings=[embedding],
            ids=[f"transcript_{i}"]
        )

# Phase 4: Query transcripts
def query_transcripts(query: str, collection: Collection, openai_client: OpenAI, top_k: int = 5) -> List[str]:
    """
    Queries the earning transcripts in the ChromaDB collection.

    Args:
        query (str): The query string.
        collection (Collection): The ChromaDB collection to query.
        openai_client (OpenAI): The OpenAI client for embedding generation.
        top_k (int): The number of top results to return.
    Returns:
        List[str]: The top_k relevant earning transcripts.
    """
    query_embedding = openai_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )

    embedding = query_embedding.data[0].embedding
    
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    
    return results['documents'][0]

# Phase 5: Generate answers
def generate_answers(query: str, relevant_transcripts: List[str], openai_client: OpenAI) -> str:
    """
    Generates answers based on the query and relevant earning transcripts.

    Args:
        query (str): The query string.
        relevant_transcripts (List[str]): The relevant earning transcripts.
        openai_client (OpenAI): The OpenAI client for answer generation.
    Returns:
        str: The generated answer.
    """
    context = "\n\n".join(relevant_transcripts)
    prompt = f"""You are a financial analyst assistant helping investors understand earnings calls.

Based on the earnings call transcript excerpts below, answer the user's question.

IMPORTANT INSTRUCTIONS:
- Focus on specific numbers, percentages, and financial metrics
- Mention growth rates and comparisons to previous periods
- Highlight what executives emphasized as most important
- Keep the answer concise and focused on what matters to investors
- If asked about revenue or highlights, prioritize: actual numbers, year-over-year growth, margins, and guidance
- Use natural conversational tone, not bullet points unless specifically asked

Context from earnings call:
{context}

Question: {query}

Answer:"""
    
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful financial analyst assistant who provides clear, concise insights from earnings calls. Focus on what matters to investors: numbers, growth rates, and forward guidance."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7
    )
    
    return response.choices[0].message.content

# Example usage
if __name__ == "__main__":

    if collection.count() == 0:
        print("Populating ChromaDB with earning transcripts...")
        # Fetch earning transcripts
        transcripts = fetch_earning_transcript()
        # Chunk transcripts
        chunked_transcripts = chunk_transcripts(transcripts)
        # Vectorize and store transcripts
        vectorize_transcripts(chunked_transcripts, collection, openai_client)
        # Query transcripts
    answer_context = ""
    while True:
        query = input("Enter your query about earning transcripts (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        query = f"{query}"
        relevant_transcripts = query_transcripts(query, collection, openai_client)
        # Generate answers
        answer = generate_answers(query, relevant_transcripts, openai_client)
        print("\n Answer to the query:"+answer+"\n")
        answer_context += f"\n\nQ: {query}\nA: {answer}\n"

