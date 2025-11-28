from typing import List
from chromadb import Collection, Client, PersistentClient
from openai import OpenAI
import os
import fileinput as f
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import hashlib
import json
from datetime import datetime
import sys

# Load environment variables from .env file
load_dotenv()

# Initialize clients
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
chroma_client = PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="earnings_transcripts")

# Metadata tracking
METADATA_FILE = "vectorization_metadata.json"

def load_metadata() -> dict:
    """Load metadata about what's been processed"""
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {'processed_files': {}, 'last_update': None, 'total_chunks': 0}

def save_metadata(metadata: dict):
    """Save metadata about processed files"""
    metadata['last_update'] = datetime.now().isoformat()
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

def get_file_hash(filepath: str) -> str:
    """Get hash of file to detect changes"""
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_available_companies(collection: Collection) -> List[str]:
    """
    Get list of unique companies in the collection.
    """
    # Get all unique companies from metadata
    all_data = collection.get()
    companies = set()
    if all_data and 'metadatas' in all_data:
        for metadata in all_data['metadatas']:
            if metadata and 'company' in metadata:
                companies.add(metadata['company'])
    
    return sorted(list(companies))

# Phase 1: Fetch earning transcripts
def fetch_earning_transcript(folder: str = 'transcripts') -> List[dict]:
    """
    Fetches earning transcripts from the transcripts folder.
    Returns only NEW or CHANGED transcripts.
    """
    metadata = load_metadata()
    processed_files = metadata.get('processed_files', {})
    
    transcripts = []
    new_count = 0
    updated_count = 0
    skipped_count = 0
    
    if not os.path.exists(folder):
        print(f"Error: Folder '{folder}' does not exist")
        return transcripts
    
    for filename in os.listdir(folder):
        if not filename.endswith('.txt'):
            continue
            
        file_path = os.path.join(folder, filename)
        file_hash = get_file_hash(file_path)
        
        # Check if file already processed and unchanged
        if filename in processed_files:
            if processed_files[filename]['hash'] == file_hash:
                skipped_count += 1
                continue
            else:
                updated_count += 1
                print(f"  ↻ Updated: {filename}")
        else:
            new_count += 1
            print(f"  + New: {filename}")
        
        # Read file
        company_name = filename.split('_')[0]
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            transcripts.append({
                'company': company_name,
                'content': content,
                'filename': filename,
                'hash': file_hash
            })
    
    print(f"\n📊 Summary: {new_count} new, {updated_count} updated, {skipped_count} skipped")
    return transcripts

# Phase 2: Chunking transcripts
def chunk_transcripts(transcripts: List[dict], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[dict]:
    """
    Chunks the earning transcripts into smaller pieces.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_transcripts = []
    for transcript in transcripts:
        chunks = text_splitter.split_text(transcript['content'])
        for i, chunk in enumerate(chunks):
            chunked_transcripts.append({
            "Company": transcript['company'],
            "Filename": transcript['filename'], 
            "Hash": transcript['hash'],
            "ChunkIndex": i,
            "Text": chunk})
    print(f"📦 Created {len(chunked_transcripts)} chunks from {len(transcripts)} transcripts")
    return chunked_transcripts

# Phase 3: Vectorize and store in ChromaDB
def vectorize_transcripts(
        chunked_transcripts: List[dict], 
        collection: Collection, 
        openai_client: OpenAI, 
        batch_size: int = 100
        ) -> int:
    """
    FAST vectorization using batch embeddings.
    
    Returns:
        Number of chunks processed
    """
    if not chunked_transcripts:
        print("No chunks to process")
        return 0

    print(f" Vectorizing {len(chunked_transcripts)} chunks (batch size: {batch_size})...")

    start_time = datetime.now()
    metadata = load_metadata()
    
    # Get current max ID to avoid collisions
    existing_count = collection.count()
    
    total_api_calls = 0


    for i in range(0, len(chunked_transcripts), batch_size):
        batch = chunked_transcripts[i:i+batch_size]
        texts = [doc['Text'] for doc in batch]


        embedding = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        total_api_calls += 1

        embeddings = [item.embedding for item in embedding.data]
        
        #Prepare data for ChromaDB
        ids = [f"chunkid_{existing_count + i + j}" for j in range(len(batch))]
        collection.add(
            documents=[doc['Text'] for doc in batch],
            embeddings=embeddings,
            ids=ids,
            metadatas=[{
                'company': doc['Company'], 
                'filename': doc['Filename'],
                'chunk_index': doc['ChunkIndex']
                    } for doc in batch])

        # Update metadata for processed files
        for doc in batch:
            if doc['Filename'] not in metadata['processed_files']:
                metadata['processed_files'][doc['Filename']] = {}
                metadata['processed_files'][doc['Filename']]['hash'] = doc['Hash']
                metadata['processed_files'][doc['Filename']]['chunks'] = metadata['processed_files'][doc['Filename']].get('chunks', 0) + 1
        
        if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(chunked_transcripts):
            processed = min(i + batch_size, len(chunked_transcripts))
            print(f"  ✓ {processed}/{len(chunked_transcripts)} chunks processed")
    # Save metadata
    metadata['total_chunks'] = collection.count()
    save_metadata(metadata)
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n Vectorization complete!")
    print(f"   Total chunks: {len(chunked_transcripts)}")
    print(f"   API calls: {total_api_calls}")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Speed: {len(chunked_transcripts)/elapsed:.1f} chunks/second")
    
    return len(chunked_transcripts)

def update_vector_database_incremental(folder: str = 'transcripts') -> bool:
    """
    Incrementally update the vector database with new transcripts only.
    This is what runs quarterly in production.
    
    Returns:
        True if successful
    """
    print("\n" + "="*60)
    print("🔄 INCREMENTAL DATABASE UPDATE")
    print("="*60 + "\n")
    
    # Load only NEW or CHANGED transcripts
    transcripts = fetch_earning_transcript(folder)
    
    if not transcripts:
        print("\n✓ Database is up to date! No new transcripts to process.")
        return True
    
    # Chunk new transcripts
    chunked = chunk_transcripts(transcripts)
    
    # Vectorize and add to existing database
    processed = vectorize_transcripts_BATCH(chunked, collection, openai_client)
    
    if processed > 0:
        print("\n" + "="*60)
        print(f"✓ Successfully added {processed} new chunks to database")
        print(f"✓ Total database size: {collection.count()} chunks")
        print("="*60)
        return True
    
    return False

# Phase 4: Query transcripts
def query_transcripts(query: str, collection: Collection, openai_client: OpenAI, company: str= None, top_k: int = 5) -> List[str]:
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
        n_results=top_k,
        where={"company": {"$eq": company}} if company else None
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


def rebuild_database_from_scratch(folder: str = 'transcripts') -> bool:
    """
    Nuclear option: Rebuild entire database.
    Only use this if you change chunking strategy or encounter corruption.
    
    WARNING: Deletes all existing data!
    """
    print("\n" + "="*60)
    print("⚠️  WARNING: FULL DATABASE REBUILD")
    print("="*60)
    
    response = input("This will delete all existing data. Continue? (yes/no): ")
    if response.lower() != 'yes':
        print("Cancelled.")
        return False
    
    # Delete existing collection
    try:
        chroma_client.delete_collection(name="earnings_transcripts")
        print("✓ Deleted existing collection")
    except:
        print("No existing collection to delete")
    
    # Create fresh collection
    global collection
    collection = chroma_client.get_or_create_collection(name="earnings_transcripts")
    
    # Reset metadata
    if os.path.exists(METADATA_FILE):
        os.remove(METADATA_FILE)
    
    # Process all transcripts
    print("\n📂 Loading all transcripts...")
    
    transcripts = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder, filename)
            company_name = filename.split('_')[0]
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                transcripts.append({
                    'company': company_name,
                    'content': content,
                    'filename': filename,
                    'hash': get_file_hash(file_path)
                })
    
    if not transcripts:
        print("Error: No transcripts found!")
        return False
    
    print(f"✓ Loaded {len(transcripts)} transcripts")
    
    # Chunk and vectorize
    chunked = chunk_transcripts(transcripts)
    processed = vectorize_transcripts(chunked, collection, openai_client)
    
    print("\n" + "="*60)
    print(f"✓ Database rebuilt with {processed} chunks")
    print("="*60)
    
    return True


def get_database_stats() -> dict:
    """Get statistics about the vector database"""
    metadata = load_metadata()
    
    return {
        'total_chunks': collection.count(),
        'total_files': len(metadata.get('processed_files', {})),
        'last_update': metadata.get('last_update', 'Never'),
        'companies': len(set(m['company'] for m in collection.get()['metadatas'] if m))
    }


def print_database_stats():
    """Print database statistics"""
    stats = get_database_stats()
    
    print("\n" + "="*60)
    print("📊 DATABASE STATISTICS")
    print("="*60)
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total files: {stats['total_files']}")
    print(f"Companies: {stats['companies']}")
    print(f"Last update: {stats['last_update']}")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":


    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "update":
            # Incremental update (production mode)
            update_vector_database_incremental()
            print_database_stats()
            sys.exit(0)
            
        elif command == "rebuild":
            # Full rebuild (use sparingly)
            rebuild_database_from_scratch()
            print_database_stats()
            sys.exit(0)
            
        elif command == "stats":
            # Show statistics
            print_database_stats()
            sys.exit(0)
            
        else:
            print("Unknown command. Use: update, rebuild, or stats")
    # else:
    #     # Default: incremental update
    #     print("Running incremental update...")
    #     update_vector_database_incremental()
    #     print_database_stats()


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

    companies = set()
    companies = get_available_companies(collection)
    available_companies = []
    for i, company in enumerate(companies):
        available_companies.append({company : i})
        print(f"[{i+1}]. {company}")

    selected_company_index = int(input("Select a company by number: ")) - 1
    selected_company_dict = available_companies[selected_company_index]
    selected_company = list(selected_company_dict.keys())[0]
    print(f"Selected company: {selected_company}")

    while True:
        query = input("Enter your query about earning transcripts (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        query = f"{query}"+f"[previous context: {answer_context}]"
        relevant_transcripts = query_transcripts(query, collection, openai_client, selected_company)
        # Generate answers
        answer = generate_answers(query, relevant_transcripts, openai_client)
        print("\n Answer to the query:"+answer+"\n")
        answer_context += f"\n\nQ: {query}\nA: {answer}\n"
