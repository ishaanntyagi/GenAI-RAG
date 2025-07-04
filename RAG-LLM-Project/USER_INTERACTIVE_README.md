# User-Interactive RAG Pipeline

This directory contains a complete user-interactive RAG (Retrieval-Augmented Generation) pipeline that allows users to upload their own PDF files and interact with them using Google's Gemini AI.

## What This Pipeline Does

1. **User Input**: Prompts for PDF file path and Google API key
2. **PDF Processing**: Extracts text from the PDF and splits it into chunks
3. **Embeddings**: Creates vector embeddings using SentenceTransformer
4. **Vector Database**: Stores embeddings in ChromaDB for efficient retrieval
5. **Q&A Interface**: Provides an interactive question-answering system
6. **AI Responses**: Uses Google Gemini to generate responses based on retrieved context

## Files

### New User-Interactive Files
- `User-Interactive-RAG-Pipeline.ipynb` - Jupyter notebook version
- `user_interactive_rag_pipeline.py` - Python script version
- `USER_INTERACTIVE_README.md` - This documentation file

### Original Project Files (Unchanged)
- `01-Data-Chunking.ipynb` - Original PDF chunking notebook
- `02-Embeddings.ipynb` - Original embeddings creation notebook
- `03-Simple-Retrieval.ipynb` - Original simple retrieval notebook
- `04-VectorDB-Chroma.ipynb` - Original ChromaDB setup notebook
- `05-LLM-API-Retrieval.ipynb` - Original LLM API retrieval notebook

## Prerequisites

### Required Python Libraries
```bash
pip install pdfplumber
pip install google-generativeai
pip install sentence-transformers
pip install chromadb
```

### Google API Key
You need a Google Gemini API key. Get one from:
https://makersuite.google.com/app/apikey

## How to Use

### Option 1: Jupyter Notebook (Recommended)
1. Open `User-Interactive-RAG-Pipeline.ipynb` in Jupyter
2. Run cells sequentially
3. When prompted, provide:
   - Full path to your PDF file
   - Your Google API key
4. Follow the interactive prompts

### Option 2: Python Script
1. Run the script:
   ```bash
   python user_interactive_rag_pipeline.py
   ```
2. Follow the interactive prompts
3. Provide PDF path and API key when requested

## Features

### 🔄 Complete Pipeline Integration
- Combines all steps from the original notebooks
- No need to run multiple files
- Streamlined user experience

### 📁 Flexible PDF Input
- Accepts any PDF file path
- Validates file existence
- Provides clear error messages

### 🔐 Secure API Key Handling
- Prompts for API key at runtime
- No hardcoded credentials
- Clear setup instructions

### 💬 Interactive Q&A
- Ask unlimited questions
- Shows retrieved context
- Option to view source chunks
- Built-in help and statistics

### 📊 System Information
- Processing progress updates
- Statistics about chunks and embeddings
- Collection information

### 🧹 Cleanup Options
- Optional temporary file removal
- Preserves or removes vector database
- User-controlled cleanup

## Usage Examples

### Sample Workflow
1. **Start the system**:
   ```
   Enter PDF path: /path/to/your/document.pdf
   Enter API key: your-google-api-key
   ```

2. **System processes the PDF**:
   ```
   ✓ PDF file found
   ✓ Extracted text from 25 pages
   ✓ Created 150 chunks
   ✓ Generated embeddings
   ✓ Added 150 documents to ChromaDB
   ```

3. **Ask questions**:
   ```
   Your question: What is the main topic of this document?
   🤖 Answer: Based on the document, the main topic is...
   ```

### Available Commands
- Type any question about your PDF
- `help` - Show available commands
- `stats` - Display system statistics
- `quit` or `exit` - Exit the program

## Technical Details

### Text Processing
- Uses `pdfplumber` for reliable PDF text extraction
- Chunks text by paragraphs with size limits (~1000 characters)
- Preserves document structure

### Embeddings
- Uses `all-MiniLM-L6-v2` SentenceTransformer model
- Same model as original notebooks for consistency
- Generates 384-dimensional embeddings

### Vector Database
- ChromaDB for efficient similarity search
- Persistent storage with unique session IDs
- Metadata includes chunk information and source

### AI Generation
- Google Gemini 1.5 Flash model
- Context-aware responses
- Error handling for API issues

## File Management

### Temporary Files Created
- `user_chunks_TIMESTAMP.pkl` - Processed text chunks
- `user_embeddings_TIMESTAMP.pkl` - Generated embeddings
- `user_chroma_db_TIMESTAMP/` - Vector database directory

### Automatic Cleanup
- Option to remove temporary files after session
- Preserves original PDF file
- User confirmation before deletion

## Differences from Original Notebooks

### Enhancements
1. **User Input**: Interactive prompts instead of hardcoded paths
2. **API Key Security**: Runtime input instead of hardcoded keys
3. **Error Handling**: Comprehensive error checking and user feedback
4. **Progress Updates**: Real-time processing status
5. **Session Management**: Unique identifiers prevent conflicts
6. **Cleanup Options**: Organized temporary file management

### Maintained Compatibility
- Same processing algorithms as original notebooks
- Identical embedding models and parameters
- Compatible vector database format
- Same LLM API usage patterns

## Troubleshooting

### Common Issues

**PDF Not Found**
```
Error: File not found at /path/to/file.pdf
```
- Check file path spelling
- Use absolute paths
- Ensure file exists and is accessible

**API Key Issues**
```
Error generating response: API key invalid
```
- Verify API key from Google AI Studio
- Check for extra spaces or characters
- Ensure API key has proper permissions

**Import Errors**
```
ModuleNotFoundError: No module named 'pdfplumber'
```
- Install required libraries: `pip install pdfplumber google-generativeai sentence-transformers chromadb`

**Memory Issues**
```
Out of memory during embedding generation
```
- Try with a smaller PDF file
- Increase system RAM
- Process in smaller batches

### Performance Tips
- Larger PDFs take longer to process (5-10 minutes for 100+ pages)
- First-time model downloads may take several minutes
- Vector database queries are fast after initial setup

## Integration Notes

This pipeline is designed to work alongside the original notebooks without conflicts:
- Uses different file naming conventions
- Creates separate vector databases
- Doesn't modify original pickle files
- Independent session management

## License and Credits

Based on the original RAG-LLM-Project notebooks with enhancements for user interaction and production use.