# Medical Chat Bot

A RAG (Retrieval-Augmented Generation) based medical Q&A chatbot that answers health-related questions using a local Llama-2-7B model and Pinecone vector database.

## Features

- **PDF-based Knowledge Base**: Processes medical textbooks (PDF format) for accurate, sourced information
- **Semantic Search**: Uses sentence-transformers embeddings for intelligent query-document matching
- **Local LLM Inference**: Runs Llama-2-7B-Chat locally via CTransformers (no API costs)
- **Vector Database**: Pinecone for fast, scalable similarity search over medical content
- **LangChain Orchestration**: Clean, modular RAG pipeline with retriever chains

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  PDF Docs   │ ──► │  Text Splitter   │ ──► │  Embeddings     │
│  (data/)    │     │  (500 chars)     │     │  (MiniLM-L6-v2) │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User      │ ◄── │   Llama-2-7B     │ ◄── │   Pinecone      │
│   Query     │     │   (Local LLM)    │     │   Vector Store  │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

## Project Structure

```
Medical Chat Bot/
├── data/                       # Medical textbook PDFs
│   └── Medical_book.pdf
├── model/                      # Llama-2 GGML model
│   └── llama-2-7b-chat.ggmlv3.q4_0.bin
├── research/                   # Development notebooks
│   └── trials.ipynb
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (API keys)
└── CLAUDE.md                   # Development guidelines
```

## Installation

### 1. Clone and Setup Virtual Environment

```bash
cd "Medical Chat Bot"
python3 -m venv .mchatbot
source .mchatbot/bin/activate  # On macOS/Linux
# or: .mchatbot\Scripts\activate  # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Llama-2 Model

Download the quantized Llama-2-7B-Chat model from HuggingFace:

```bash
# Manual download:
# Visit: https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
# Download: llama-2-7b-chat.ggmlv3.q4_0.bin
# Place in: model/
```

Or use `wget`/`curl`:

```bash
cd model
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q4_0.bin
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
PINECONE_API_KEY=your_api_key_here
PINECONE_ENV=us-east-1
```

Get your Pinecone API key from [Pinecone Console](https://app.pinecone.io/).

## Usage

### Running the Notebook

The main development and testing workflow is in the Jupyter notebook:

```bash
jupyter notebook research/trials.ipynb
```

### Running as a Script

To convert the notebook to a Python script:

```bash
jupyter nbconvert --to script research/trials.ipynb
python research/trials.py
```

### Example Query

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# Setup retriever
retriever = docsearch.as_retriever(search_kwargs={"k": 2})

# Create chain
template = """You are a medical expert. Answer based on the context.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Query
result = chain.invoke("What causes diabetes?")
print(result)
```

## Configuration

### Text Splitting
- **Chunk size**: 500 characters
- **Chunk overlap**: 20 characters
- Adjust in `RecursiveCharacterTextSplitter` for different granularity

### LLM Settings
```python
config = {
    'max_new_tokens': 512,
    'temperature': 0.8
}
```

### Embeddings
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Fast, lightweight, suitable for semantic search

### Pinecone Index
- Index name: `medical-chatbot`
- Environment: `us-east-1` (configurable in `.env`)

## Dependencies

| Package | Purpose |
|---------|---------|
| langchain | RAG orchestration framework |
| langchain-community | Community integrations (Pinecone, CTransformers) |
| langchain-text-splitters | Document chunking |
| langchain-pinecone | Pinecone vector store integration |
| ctransformers | Local GGML model inference |
| sentence-transformers | Text embeddings |
| pinecone | Vector database client |
| flask | Web server (optional) |
| python-dotenv | Environment variable management |

## Troubleshooting

### Token Limit Warnings
If you see `Number of tokens exceeded maximum context length`:
- Reduce `max_new_tokens` in LLM config
- Decrease chunk size in text splitter
- Reduce `k` in retriever search_kwargs

### Model Loading Issues
Ensure the model file exists and has correct permissions:
```bash
ls -lh model/llama-2-7b-chat.ggmlv3.q4_0.bin
# Should be ~3.5 GB
```

### Pinecone Connection Errors
- Verify API key in `.env`
- Check index exists: `pc.list_indexes()`
- Ensure environment matches your Pinecone configuration

## License

This project is for educational purposes. The Llama-2 model is subject to Meta's [Llama 2 Community License](https://ai.meta.com/llama/license/).

## References

- [LangChain Documentation](https://python.langchain.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Llama-2-7B-Chat-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)
- [Sentence Transformers](https://www.sbert.net/)