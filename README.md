# langchain-lambdadb

This package contains the LangChain integration with LambdaDB vector store.

## Installation

```bash
pip install -U langchain-lambdadb
```

## Prerequisites

Before using this integration, you need to:

1. Create a collection in LambdaDB with proper vector and text indexes
2. Have your LambdaDB credentials ready

### Creating a Collection

Create a collection in LambdaDB with the required indexes:

```python
from lambdadb import LambdaDB, models

client = LambdaDB(
    server_url="<your-project-url>",
    project_api_key="<your-project-api-key>"

)

# Create collection with vector and text indexes
client.collections.create(
    collection_name="my_collection",
    index_configs={
        "vector": {
            "type": models.TypeVector.VECTOR,
            "dimensions": 1536,  # Match your embedding dimensions
            "similarity": models.Similarity.COSINE
        },
        "text": {
            "type": models.TypeText.TEXT,
            "analyzers": [models.Analyzer.ENGLISH]
        }
    }
)
```

## Quick Start

```python
import os
from lambdadb import LambdaDB
from langchain_lambdadb import LambdaDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# Set up LambdaDB client
client = LambdaDB(
    server_url=os.getenv("LAMBDADB_PROJECT_URL"),
    project_api_key=os.getenv("LAMBDADB_PROJECT_API_KEY")
)

# Connect to existing collection
vector_store = LambdaDBVectorStore(
    client=client,
    collection_name="my_collection",  # Must be an existing collection
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
export LAMBDADB_PROJECT_URL="<your-project-url>"
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

```python
# Search with metadata filters
filtered_results = vector_store.similarity_search(
    query="database",
    k=5,
    filter={"source": "documentation"}
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
export LAMBDADB_PROJECT_URL="<your-project-url>"
export LAMBDADB_PROJECT_API_KEY="<your-project-api-key>"
# Optional: Use existing collection instead of creating test collections
export LAMBDADB_COLLECTION_NAME="your-test-collection"
make integration_tests

# Lint and format code
make lint
make format
```
