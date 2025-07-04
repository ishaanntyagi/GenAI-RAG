#!/usr/bin/env python3
"""
User-Interactive RAG Pipeline

This script provides a complete end-to-end RAG (Retrieval-Augmented Generation) pipeline that:
1. Accepts user input for PDF file path
2. Asks for Google API key
3. Follows the same pipeline as the existing notebooks
4. Provides an interactive Q&A interface

Required Libraries:
- pdfplumber
- google-generativeai
- sentence-transformers
- chromadb
- pickle
- os
- warnings

Author: Based on the existing RAG-LLM-Project notebooks
"""

import pdfplumber
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
import pickle
import os
import warnings
from datetime import datetime
import shutil

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print('='*50)

def print_success(message):
    """Print a success message."""
    print(f"✓ {message}")

def print_error(message):
    """Print an error message."""
    print(f"❌ {message}")

def get_user_inputs():
    """Get PDF path and API key from user."""
    print_header("USER INPUT SECTION")
    
    # Get PDF file path
    print("\n--- PDF File Input ---")
    while True:
        pdf_path = input("Please enter the full path to your PDF file: ").strip()
        
        if not pdf_path:
            print_error("Path cannot be empty. Please try again.")
            continue
            
        if not os.path.exists(pdf_path):
            print_error(f"File not found at {pdf_path}")
            print("Please check the path and try again.")
            continue
            
        if not pdf_path.lower().endswith('.pdf'):
            print("Warning: The file doesn't appear to be a PDF. Proceeding anyway...")
            
        print_success(f"PDF file found: {pdf_path}")
        break
    
    # Get Google API key
    print("\n--- Google API Key Input ---")
    print("Please enter your Google Gemini API key.")
    print("You can get one from: https://makersuite.google.com/app/apikey")
    
    while True:
        api_key = input("Enter your Google API key: ").strip()
        
        if not api_key:
            print_error("API key cannot be empty!")
            continue
            
        print_success("API key received")
        break
    
    return pdf_path, api_key

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF file."""
    print_header("PDF TEXT EXTRACTION")
    
    all_text = " "
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Processing {total_pages} pages...")
            
            for i, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    all_text += page_text + "\\n"
                    if i % 10 == 0:  # Progress update every 10 pages
                        print(f"Processed {i}/{total_pages} pages")
                        
        print_success(f"Successfully extracted text from {total_pages} pages")
        print(f"Total text length: {len(all_text)} characters")
        
        return all_text
        
    except Exception as e:
        print_error(f"Error extracting text from PDF: {str(e)}")
        raise

def chunk_text(text, max_chunk_size=1000):
    """Split text into chunks of approximately max_chunk_size characters."""
    chunks = []
    
    # Split by double newlines first (paragraphs)
    paragraphs = text.split('\\n\\n')
    
    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed max_chunk_size, save current chunk
        if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = paragraph
        else:
            current_chunk += (" " if current_chunk else "") + paragraph
    
    # Add the last chunk if it exists
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def process_text_chunks(all_text):
    """Process text into chunks."""
    print_header("TEXT CHUNKING")
    
    chunks = chunk_text(all_text)
    print_success(f"Created {len(chunks)} chunks")
    print(f"Average chunk size: {sum(len(chunk) for chunk in chunks) // len(chunks)} characters")
    
    if chunks:
        print(f"\\nSample chunk (first 200 chars): {chunks[0][:200]}...")
    
    return chunks

def create_embeddings(chunks):
    """Create embeddings for text chunks."""
    print_header("EMBEDDING GENERATION")
    print("Loading SentenceTransformer model...")
    
    # Load the same model as used in the original notebooks
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print_success("SentenceTransformer model loaded")
    
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks)
    print_success(f"Generated embeddings with shape: {embeddings.shape}")
    
    return model, embeddings

def setup_vector_database(chunks, embeddings, pdf_path):
    """Setup ChromaDB vector database."""
    print_header("VECTOR DATABASE SETUP")
    
    # Create unique identifiers for this session
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    db_path = f"user_chroma_db_{timestamp}"
    collection_name = f"user_collection_{timestamp}"
    
    # Setup ChromaDB
    chroma_client = chromadb.PersistentClient(path=db_path)
    print_success(f"ChromaDB client created with path: {db_path}")
    
    # Create collection
    collection = chroma_client.get_or_create_collection(name=collection_name)
    print_success(f"Collection '{collection_name}' created")
    
    # Add documents to the collection
    print(f"Adding {len(chunks)} documents to the collection...")
    
    # Prepare data for ChromaDB
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": i, "source": os.path.basename(pdf_path)} for i in range(len(chunks))]
    
    # Add to collection
    collection.add(
        embeddings=embeddings.tolist(),
        documents=chunks,
        ids=ids,
        metadatas=metadatas
    )
    
    print_success(f"Successfully added {collection.count()} documents to ChromaDB")
    
    return collection, db_path, timestamp

def answer_question(question, model, collection, num_results=3):
    """Answer a question using the RAG pipeline."""
    
    # Generate embedding for the question
    question_embedding = model.encode(question)
    
    # Query the vector database
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=num_results
    )
    
    # Get the retrieved chunks
    retrieved_chunks = results['documents'][0]
    
    # Create context from retrieved chunks
    context = "\\n---\\n".join(retrieved_chunks)
    
    # Create prompt for the LLM
    prompt = f"""Context: {context}

Question: {question}

Answer:"""
    
    # Generate response using Google Gemini
    try:
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        response = gemini_model.generate_content(prompt)
        
        return {
            'answer': response.text,
            'retrieved_chunks': retrieved_chunks,
            'context': context
        }
    except Exception as e:
        return {
            'error': f"Error generating response: {str(e)}",
            'retrieved_chunks': retrieved_chunks,
            'context': context
        }

def interactive_qa_session(model, collection, pdf_path, chunks):
    """Run interactive Q&A session."""
    print_header("INTERACTIVE Q&A SESSION")
    print("Ask questions about your PDF document. Type 'quit' to exit.")
    print("Type 'help' for available commands.")
    print("-" * 50)
    
    while True:
        question = input("\\nYour question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Thank you for using the RAG pipeline!")
            break
        
        if question.lower() == 'help':
            print("Available commands:")
            print("- Type any question about your PDF document")
            print("- 'quit' or 'exit' or 'q' to exit")
            print("- 'help' to see this message")
            print("- 'stats' to see system statistics")
            continue
        
        if question.lower() == 'stats':
            print(f"System Statistics:")
            print(f"- PDF file: {os.path.basename(pdf_path)}")
            print(f"- Total chunks: {len(chunks)}")
            print(f"- Vector database: {collection.count()} documents")
            continue
        
        if not question:
            print("Please enter a question.")
            continue
        
        print(f"\\nProcessing question: {question}")
        print("Searching for relevant information...")
        
        # Get answer
        result = answer_question(question, model, collection)
        
        if 'error' in result:
            print_error(f"Error: {result['error']}")
            print("\\nRetrieved context for reference:")
            print(result['context'][:500] + "...")
        else:
            print(f"\\n🤖 Answer: {result['answer']}")
            
            # Optionally show retrieved chunks
            show_chunks = input("\\nShow retrieved chunks? (y/n): ").strip().lower()
            if show_chunks == 'y':
                print("\\n📄 Retrieved chunks:")
                for i, chunk in enumerate(result['retrieved_chunks'], 1):
                    print(f"\\nChunk {i}:")
                    print(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        
        print("-" * 50)

def run_sample_questions(model, collection):
    """Test the system with sample questions."""
    print_header("SAMPLE QUESTIONS TEST")
    
    sample_questions = [
        "What is the main topic of this document?",
        "Can you summarize the key points?",
        "What are the most important concepts mentioned?"
    ]
    
    for i, question in enumerate(sample_questions, 1):
        print(f"\\n{i}. Question: {question}")
        print("   Processing...")
        
        result = answer_question(question, model, collection)
        
        if 'error' in result:
            print(f"   ❌ Error: {result['error']}")
        else:
            print(f"   🤖 Answer: {result['answer'][:200]}...")
        
        print("-" * 40)

def cleanup_files(db_path, timestamp):
    """Clean up temporary files."""
    print_header("CLEANUP")
    
    cleanup = input("Do you want to remove temporary files? (y/n): ").strip().lower()
    
    if cleanup == 'y':
        try:
            # Remove pickle files
            chunks_filename = f"user_chunks_{timestamp}.pkl"
            embeddings_filename = f"user_embeddings_{timestamp}.pkl"
            
            if os.path.exists(chunks_filename):
                os.remove(chunks_filename)
                print_success(f"Removed {chunks_filename}")
            
            if os.path.exists(embeddings_filename):
                os.remove(embeddings_filename)
                print_success(f"Removed {embeddings_filename}")
            
            # Remove ChromaDB directory
            if os.path.exists(db_path):
                shutil.rmtree(db_path)
                print_success(f"Removed {db_path}")
            
            print_success("Cleanup completed")
        except Exception as e:
            print_error(f"Error during cleanup: {str(e)}")
    else:
        print("Temporary files preserved:")
        print(f"- ChromaDB: {db_path}")

def main():
    """Main function to run the RAG pipeline."""
    print("=" * 60)
    print("           USER-INTERACTIVE RAG PIPELINE")
    print("=" * 60)
    print("This pipeline will process your PDF and enable Q&A functionality.")
    print("Please ensure you have all required libraries installed.")
    
    try:
        # Step 1: Get user inputs
        pdf_path, api_key = get_user_inputs()
        
        # Configure Google AI API
        genai.configure(api_key=api_key)
        print_success("Google AI API configured successfully")
        
        # Step 2: Extract text from PDF
        all_text = extract_text_from_pdf(pdf_path)
        
        # Step 3: Process text into chunks
        chunks = process_text_chunks(all_text)
        
        # Step 4: Create embeddings
        model, embeddings = create_embeddings(chunks)
        
        # Step 5: Setup vector database
        collection, db_path, timestamp = setup_vector_database(chunks, embeddings, pdf_path)
        
        # Step 6: Test with sample questions (optional)
        test_samples = input("\\nWould you like to test with sample questions first? (y/n): ").strip().lower()
        if test_samples == 'y':
            run_sample_questions(model, collection)
        
        # Step 7: Interactive Q&A session
        interactive_qa_session(model, collection, pdf_path, chunks)
        
        # Step 8: Cleanup
        cleanup_files(db_path, timestamp)
        
    except KeyboardInterrupt:
        print("\\n\\nProgram interrupted by user.")
    except Exception as e:
        print_error(f"An unexpected error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()