"""
Earnings Call Intelligence - Streamlit App
Simple UI for querying earnings call transcripts
"""

import streamlit as st
from typing import List, Dict

# Import from your RAG implementation
# Make sure earnings_call_transcript.py is in the same directory
# Or adjust the import path as needed
try:
    from earnings_call_transcript import (
        collection,
        openai_client,
        query_transcripts,
        generate_answers,
        get_available_companies,
        get_database_stats
    )
except ImportError:
    st.error("⚠️ Could not import RAG functions. Make sure earnings_call_transcript.py is in the same directory.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Earnings Call Intelligence",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Earnings Call Intelligence")
st.write("Ask questions about quarterly earnings calls from major companies")

# Sidebar - Database stats
with st.sidebar:
    st.header("📈 Database Info")
    
    try:
        stats = get_database_stats()
        st.metric("Total Companies", stats['companies'])
        st.metric("Total Transcripts", stats['total_files'])
        st.metric("Total Chunks", stats['total_chunks'])
        st.caption(f"Last updated: {stats['last_update']}")
    except Exception as e:
        st.error("Could not load database stats")
        st.caption(str(e))
    
    st.write("---")
    
    # Instructions
    st.subheader("💡 How to use")
    st.write("""
    1. Select a company
    2. Ask a question about their earnings
    3. Get instant AI-powered answers
    
    **Example questions:**
    - What was the revenue?
    - How did they perform vs last quarter?
    - What did management say about guidance?
    """)
    
    st.write("---")
    st.caption("Built with OpenAI + ChromaDB")

# Main content
col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("🏢 Select Company")
    
    # Get available companies
    try:
        companies = get_available_companies(collection)
        
        if not companies:
            st.error("No companies found in database!")
            st.info("Run: Build the Vector Store with earnings_call_transcripts. eg., python scripts/build_vector_store.py")
            st.stop()
        
        # Add "All Companies" option
        company_options = ["All Companies"] + companies
        
        selected_company = st.selectbox(
            "Choose a company:",
            company_options,
            key="company_select"
        )
        
        # Show company count
        st.caption(f"{len(companies)} companies available")
        
    except Exception as e:
        st.error(f"Error loading companies: {e}")
        st.stop()

with col2:
    st.subheader("💬 Ask Questions")
    
    # Initialize conversation history in session state
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if query := st.chat_input("Ask about earnings calls..."):
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Add to messages
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Process query
        with st.chat_message("assistant"):
            with st.spinner("Searching earnings calls..."):
                try:
                    # Determine company filter
                    company_filter = None if selected_company == "All Companies" else selected_company
                    
                    # Query transcripts
                    relevant_transcripts = query_transcripts(
                        query, 
                        collection, 
                        openai_client, 
                        company_filter,
                        top_k=3
                    )
                    
                    # Generate answer with conversation history
                    answer = generate_answers(
                        query,
                        relevant_transcripts['documents'],
                        openai_client,
                        st.session_state.conversation_history,
                        max_history=3
                    )
                    
                    # Display answer
                    st.write(answer)
                    
                    # Add to conversation history
                    st.session_state.conversation_history.append({
                        'question': query,
                        'answer': answer
                    })
                    
                    # Add to messages
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    # Show sources in expander
                    with st.expander("📄 View Source Excerpts"):
                        for i, (doc,meta) in enumerate(zip(relevant_transcripts['documents'], relevant_transcripts['metadatas']), 1):
                            st.text_area(
                                f"**Source {i}:** `{meta.get('filename','N/A')}` (Chunk_Index: {meta.get('chunk_index','N/A')})",
                                doc[:500] + "..." if len(doc) > 500 else doc,
                                height=100,
                                key=f"source_{i}_{len(st.session_state.messages)}"
                            )
                
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
                    st.exception(e)

# Clear conversation button
if st.session_state.messages:
    if st.button("🔄 Clear Conversation", key="clear"):
        st.session_state.conversation_history = []
        st.session_state.messages = []
        st.rerun()

# Footer
st.write("---")
st.caption(" Powered by GPT-4o-mini + ChromaDB | Developed by Abdullah Syedali ")