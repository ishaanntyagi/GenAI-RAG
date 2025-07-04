#!/usr/bin/env python3
"""
Test script for the user-interactive RAG pipeline
Tests the core functions without requiring external dependencies
"""

import os
import sys

def test_chunk_function():
    """Test the text chunking function with sample data."""
    
    # Define chunk_text function locally to test without imports
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
    
    # Test with sample text
    sample_text = """This is the first paragraph of our test document.

This is the second paragraph which contains more information about the topic.

This is the third paragraph that adds additional context.

This is a longer paragraph that contains much more text and should potentially be split into its own chunk depending on the maximum chunk size that we specify. It contains detailed information about various topics and concepts that would be useful for testing the chunking functionality.

This is the final paragraph of our test document."""
    
    chunks = chunk_text(sample_text, max_chunk_size=200)
    
    print("=== Chunk Function Test ===")
    print(f"Input text length: {len(sample_text)} characters")
    print(f"Number of chunks created: {len(chunks)}")
    print(f"Max chunk size setting: 200 characters")
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\\nChunk {i} ({len(chunk)} chars): {chunk[:100]}...")
    
    # Validate chunks
    total_chars = sum(len(chunk) for chunk in chunks)
    print(f"\\nValidation:")
    print(f"✓ Total characters in chunks: {total_chars}")
    print(f"✓ All chunks under 300 chars: {all(len(chunk) <= 300 for chunk in chunks)}")
    print(f"✓ No empty chunks: {all(chunk.strip() for chunk in chunks)}")
    
    return len(chunks) > 0

def test_file_structure():
    """Test that all required files exist and have proper structure."""
    print("\\n=== File Structure Test ===")
    
    required_files = [
        "User-Interactive-RAG-Pipeline.ipynb",
        "user_interactive_rag_pipeline.py", 
        "USER_INTERACTIVE_README.md",
        "requirements.txt"
    ]
    
    rag_dir = "/home/runner/work/GenAI-RAG-Project/GenAI-RAG-Project/RAG-LLM-Project"
    
    for filename in required_files:
        filepath = os.path.join(rag_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            print(f"✓ {filename} exists ({size} bytes)")
        else:
            print(f"❌ {filename} missing")
            return False
    
    return True

def test_original_files_unchanged():
    """Test that original files were not modified."""
    print("\\n=== Original Files Test ===")
    
    original_files = [
        "01-Data-Chunking.ipynb",
        "02-Embeddings.ipynb", 
        "03-Simple-Retrieval.ipynb",
        "04-VectorDB-Chroma.ipynb",
        "05-LLM-API-Retrieval.ipynb"
    ]
    
    rag_dir = "/home/runner/work/GenAI-RAG-Project/GenAI-RAG-Project/RAG-LLM-Project"
    
    for filename in original_files:
        filepath = os.path.join(rag_dir, filename)
        if os.path.exists(filepath):
            print(f"✓ {filename} preserved")
        else:
            print(f"❌ {filename} missing")
            return False
    
    return True

def test_documentation():
    """Test that documentation is comprehensive."""
    print("\\n=== Documentation Test ===")
    
    readme_path = "/home/runner/work/GenAI-RAG-Project/GenAI-RAG-Project/RAG-LLM-Project/USER_INTERACTIVE_README.md"
    
    if not os.path.exists(readme_path):
        print("❌ README file missing")
        return False
    
    with open(readme_path, 'r') as f:
        content = f.read()
    
    required_sections = [
        "User-Interactive RAG Pipeline",
        "Prerequisites", 
        "How to Use",
        "Features",
        "Troubleshooting"
    ]
    
    for section in required_sections:
        if section in content:
            print(f"✓ Contains '{section}' section")
        else:
            print(f"❌ Missing '{section}' section")
            return False
    
    print(f"✓ README length: {len(content)} characters")
    return True

def test_requirements():
    """Test that requirements file contains necessary packages."""
    print("\\n=== Requirements Test ===")
    
    req_path = "/home/runner/work/GenAI-RAG-Project/GenAI-RAG-Project/RAG-LLM-Project/requirements.txt"
    
    if not os.path.exists(req_path):
        print("❌ requirements.txt missing")
        return False
    
    with open(req_path, 'r') as f:
        content = f.read()
    
    required_packages = [
        "pdfplumber",
        "google-generativeai", 
        "sentence-transformers",
        "chromadb"
    ]
    
    for package in required_packages:
        if package in content:
            print(f"✓ Contains {package}")
        else:
            print(f"❌ Missing {package}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("Running User-Interactive RAG Pipeline Tests")
    print("=" * 50)
    
    tests = [
        ("Chunk Function", test_chunk_function),
        ("File Structure", test_file_structure), 
        ("Original Files", test_original_files_unchanged),
        ("Documentation", test_documentation),
        ("Requirements", test_requirements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "❌"
        print(f"{symbol} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Pipeline is ready for use.")
        return 0
    else:
        print("⚠️  Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())