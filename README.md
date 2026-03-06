# langchain-lambdadb

This package contains the LangChain integration with LambdaDB vector store.

## Installation

```bash
pip install -U langchain-lambdadb
```

## Prerequisites

- LambdaDB credentials (API key). The collection is created automatically if it does not exist (with a default vector and text index), unless you set `create_if_not_exists=False`.

To filter by metadata, either create the collection beforehand with `index_configs` that include those metadata fields, or pass `index_configs` (and optionally `partition_config`) when constructing the vector store; in LambdaDB, only fields listed in `index_configs` can be used as filters.

### Optional: Creating a collection manually

You can create a collection in LambdaDB yourself (e.g. to define custom indexes or partitions):

```python
from lambdadb import LambdaDB, models

client = LambdaDB(
    base_url="https://api.lambdadb.ai",
    project_name="playground",
    project_api_key="<your-project-api-key>",
)

client.collections.create(
    collection_name="my_collection",
    index_configs={
        "vector": {
            "type": models.TypeVector.VECTOR,
            "dimensions": 1536,  # Match your embedding dimensions
            "similarity": models.Similarity.COSINE
        },
        "page_content": {
            "type": models.TypeText.TEXT,
            "analyzers": [models.Analyzer.ENGLISH]
        }
    }
)
```

The index name for the text field (e.g. `"page_content"`) must match the `text_field` argument when constructing `LambdaDBVectorStore` (default is `"page_content"`). If you use a different name (e.g. `"text"`), pass it as `text_field=...`.

## Quick Start

```python
import os
from lambdadb import LambdaDB
from langchain_lambdadb import LambdaDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Set up LambdaDB client (0.7.0+ use base_url and project_name)
client = LambdaDB(
    base_url=os.getenv("LAMBDADB_BASE_URL", "https://api.lambdadb.ai"),
    project_name=os.getenv("LAMBDADB_PROJECT_NAME", "playground"),
    project_api_key=os.getenv("LAMBDADB_PROJECT_API_KEY"),
)

# Uses existing collection or creates it if missing
vector_store = LambdaDBVectorStore(
    client=client,
    collection_name="my_collection",
    embedding=OpenAIEmbeddings()
)

# Add documents
documents = [
    Document(page_content="LambdaDB is a vector database", metadata={"source": "docs"}),
    Document(page_content="LangChain integrates with LambdaDB", metadata={"source": "docs"}),
]
vector_store.add_documents(documents)

# Search for similar documents
results = vector_store.similarity_search("What is LambdaDB?", k=2)
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}")
```

## Configuration

Set the following environment variables:

```bash
export LAMBDADB_BASE_URL="https://api.lambdadb.ai"   # optional, this is the default
export LAMBDADB_PROJECT_NAME="playground"             # optional, this is the default
export LAMBDADB_PROJECT_API_KEY="<your-project-api-key>"
```

## Vector Store Features

The `LambdaDBVectorStore` supports:

- **Document Operations**: Add, update, and delete documents
- **Similarity Search**: Find similar documents using vector search
- **Metadata Filtering**: Filter search results by document metadata
- **Batch Operations**: Efficient bulk document processing
- **Async Support**: Full async/await support for all operations

## Advanced Usage

### Similarity Search with Scores

```python
# Get similarity scores with results
results_with_scores = vector_store.similarity_search_with_score(
    query="vector database features",
    k=3
)

for doc, score in results_with_scores:
    print(f"Score: {score:.4f}")
    print(f"Content: {doc.page_content}")
```

### Metadata Filtering

Only metadata fields that are included in the collection's `index_configs` can be used as filters. If you use the default auto-created collection (no custom `index_configs`), add documents and use filters only on fields you passed in `index_configs` when creating the vector store.

```python
# Search with metadata filters (collection must have index_configs for "source")
filtered_results = vector_store.similarity_search(
    query="database",
    k=5,
    filter={"queryString": {"query": "source:documentation"}}
)
```

### Using as a Retriever

```python
# Use as a retriever for RAG applications
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

relevant_docs = retriever.invoke("How does LambdaDB work?")
```

## Development

For development and testing:

```bash
# Clone the repository
git clone <repository-url>
cd langchain-lambdadb

# Install with development dependencies
poetry install --with test,lint

# Run tests with mock data
make test

# Run integration tests with real LambdaDB (requires credentials)
export LAMBDADB_BASE_URL="https://api.lambdadb.ai"
export LAMBDADB_PROJECT_NAME="playground"
export LAMBDADB_PROJECT_API_KEY="<your-project-api-key>"
# Optional: Use existing collection instead of creating test collections
export LAMBDADB_COLLECTION_NAME="your-test-collection"
make integration_tests

# Lint and format code
make lint
make format
```
