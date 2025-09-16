# Chain Semantic Splitter for LangChain

`chain-semantic-splitter` is a Python library that provides an advanced `TextSplitter` for the LangChain ecosystem. Unlike traditional splitters that rely on fixed character counts, `SemanticCharacterTextSplitter` uses a powerful language model (like Google's Gemini) to understand the text and create chunks that are semantically coherent.

This approach results in more meaningful and contextually relevant text segments, which is ideal for RAG (Retrieval-Augmented Generation) pipelines and other LLM-based applications.

## Features

- **Semantic Chunking**: Merges text chunks based on their meaning, not arbitrary length.
- **LangChain Integration**: Inherits from `langchain.text_splitter.TextSplitter` for seamless use in any LangChain project.
- **Powered by LLMs**: Uses a configurable language model (e.g., `langchain_google_genai`) to make intelligent splitting decisions.
- **Robust & Reliable**: Includes built-in retry logic and fallback mechanisms for API calls.
- **Structured Decision Making**: Leverages structured JSON output for clear and debuggable merge decisions.

## Installation

You can install the library from PyPI:

```bash
pip install chain-semantic-splitter
```

*(Note: This assumes the package is published on PyPI with the name specified in `pyproject.toml`.)*

## Quick Start

Here's how to use the `SemanticCharacterTextSplitter` to split a document.

```python
from chain_semantic_splitter import SemanticCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

# 1. Initialize the LLM you want to use for decision making
# Make sure you have your GOOGLE_API_KEY set in your environment
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

# 2. Create an instance of the splitter
semantic_splitter = SemanticCharacterTextSplitter(
    llm=llm,
    chunk_size=500,      # Size of the initial chunks
    chunk_overlap=100    # Overlap for context
)

# 3. Load your document text
with open("my_document.txt", "r", encoding="utf-8") as f:
    document_text = f.read()

# 4. Split the text
# The splitter returns a list of strings
semantic_chunks_text = semantic_splitter.split_text(document_text)

# Or, to get Document objects directly:
documents = semantic_splitter.create_documents([document_text])


# 5. View the results
for i, chunk in enumerate(documents):
    print(f"--- Chunk {i+1} (Length: {len(chunk.page_content)}) ---")
    print(chunk.page_content[:250] + "...")
    print()

```

## How It Works

1.  **Pre-splitting**: The document is first split into smaller, overlapping chunks using a traditional `RecursiveCharacterTextSplitter`.
2.  **Iterative Merging**: The splitter then iterates through these initial chunks, comparing adjacent pairs.
3.  **LLM-Powered Decision**: For each pair, it asks the configured LLM whether the two chunks are semantically related and should be merged. The LLM's response is a structured JSON object (`{"should_merge": true/false, "reason": "..."}`).
4.  **Building Final Chunks**: If the LLM decides to merge, the chunks are combined. If not, the current chunk is considered complete, and a new one is started.
5.  **Final Output**: The process continues until all initial chunks have been processed, resulting in a list of semantically coherent text segments.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.
