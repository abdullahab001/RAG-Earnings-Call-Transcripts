# scripts/test_embeddings_chromadb.py

import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

def test_openai_embeddings_with_chromadb():
    """Test creating OpenAI embeddings and storing in ChromaDB"""
    
    print("🔄 Testing OpenAI Embeddings + ChromaDB Integration...\n")
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not found")
        return False
    
    openai_client = OpenAI()
    
    # Initialize ChromaDB
    chroma_client = chromadb.PersistentClient(path="./data/embeddings/test_openai")
    
    # Create collection
    collection = chroma_client.get_or_create_collection(
        name="stock_news_test",
        metadata={"description": "Test stock news with OpenAI embeddings"}
    )
    
    print("✅ ChromaDB collection created")
    
    # Sample stock news
    news_items = [
        {
            "id": "news_1",
            "text": "Apple reports record Q4 earnings, stock surges 8%",
            "ticker": "AAPL",
            "date": "2024-11-20"
        },
        {
            "id": "news_2", 
            "text": "Microsoft unveils new AI-powered Office features",
            "ticker": "MSFT",
            "date": "2024-11-21"
        },
        {
            "id": "news_3",
            "text": "Tesla stock drops 5% on production concerns",
            "ticker": "TSLA",
            "date": "2024-11-22"
        },
        {
            "id": "news_4",
            "text": "Apple announces new partnership with healthcare providers",
            "ticker": "AAPL",
            "date": "2024-11-23"
        }
    ]
    
    print(f"📰 Processing {len(news_items)} news items...")
    
    # Generate embeddings using OpenAI
    texts = [item['text'] for item in news_items]
    
    try:
        response = openai_client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        
        embeddings = [item.embedding for item in response.data]
        
        print(f"✅ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0])}")
        print(f"   Cost: ~${len(texts) * 0.00002:.6f}")
        
        # Add to ChromaDB
        collection.add(
            ids=[item['id'] for item in news_items],
            embeddings=embeddings,
            documents=texts,
            metadatas=[{
                "ticker": item['ticker'],
                "date": item['date']
            } for item in news_items]
        )
        
        print(f"✅ Stored {len(news_items)} items in ChromaDB")
        
        # Test semantic search
        print("\n🔍 Testing Semantic Search:\n")
        
        queries = [
            "What happened with Apple?",
            "Tell me about AI announcements",
            "Any production issues?"
        ]
        
        for query in queries:
            # Generate query embedding
            query_response = openai_client.embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = query_response.data[0].embedding
            
            # Search ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=2,
                include=["documents", "metadatas", "distances"]
            )
            
            print(f"Query: '{query}'")
            print(f"Top result: {results['documents'][0][0]}")
            print(f"  Ticker: {results['metadatas'][0][0]['ticker']}")
            print(f"  Similarity: {1 - results['distances'][0][0]:.3f}")
            print()
        
        # Cleanup
        chroma_client.delete_collection("stock_news_test")
        print("✅ Test collection deleted (cleanup)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("   OPENAI EMBEDDINGS + CHROMADB TEST")
    print("="*60)
    
    success = test_openai_embeddings_with_chromadb()
    
    print("\n" + "="*60)
    if success:
        print("✅ Full integration working!")
        print("\nYou've successfully:")
        print("   • Generated OpenAI embeddings")
        print("   • Stored them in ChromaDB")
        print("   • Performed semantic search")
        print("\n🎉 You're ready to build your RAG system!")
    else:
        print("❌ Integration test failed")
    print("="*60)