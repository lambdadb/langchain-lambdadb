import os
import time
import uuid
from typing import Generator

import pytest
from lambdadb import LambdaDB, models
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_lambdadb.vectorstores import LambdaDBVectorStore


class TestLambdaDBVectorStore(VectorStoreIntegrationTests):
    """Integration tests for LambdaDB vector store.

    To run with real LambdaDB service, set these environment variables:
    - LAMBDADB_PROJECT_URL: LambdaDB service endpoint
    - LAMBDADB_PROJECT_API_KEY: Your LambdaDB API key
    """

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        # Initialize LambdaDB client with environment credentials
        client = LambdaDB(
            server_url=os.getenv("LAMBDADB_PROJECT_URL"),
            project_api_key=os.getenv("LAMBDADB_PROJECT_API_KEY"),
        )

        # Use environment variable for collection name, or generate unique one for testing
        if os.getenv("LAMBDADB_COLLECTION_NAME"):
            collection_name = os.getenv("LAMBDADB_COLLECTION_NAME")
        else:
            # Generate unique collection name (max 52 chars for LambdaDB)
            # Use short UUID (8 chars) + shorter timestamp (last 6 digits)
            import time

            timestamp = int(time.time()) % 1000000  # last 6 digits
            short_uuid = uuid.uuid4().hex[:8]
            collection_name = f"test_{short_uuid}_{timestamp}"

        # Only create collection if it doesn't exist and no env var is set
        if not os.getenv("LAMBDADB_COLLECTION_NAME"):
            try:
                # Get embedding dimension from the test embeddings
                embeddings = self.get_embeddings()
                test_vector = embeddings.embed_query("test")
                dimension = len(test_vector)

                client.collections.create(
                    collection_name=collection_name,
                    index_configs={  # type: ignore[arg-type]
                        "vector": {
                            "type": models.TypeVector.VECTOR,
                            "dimensions": dimension,
                            "similarity": models.Similarity.COSINE,
                        },
                        "text": {
                            "type": models.TypeText.TEXT,
                            "analyzers": [models.Analyzer.ENGLISH],
                        },
                    },
                )

                # Wait for collection to be ready
                max_wait_time = 30  # seconds
                wait_interval = 1  # seconds
                waited = 0

                while waited < max_wait_time:
                    try:
                        collection_info = client.collections.get(
                            collection_name=collection_name
                        )
                        if (
                            collection_info.collection.collection_status.value
                            == "ACTIVE"
                        ):
                            break
                    except Exception:
                        pass
                    time.sleep(wait_interval)
                    waited += wait_interval

            except Exception as e:
                # Only ignore if collection already exists, otherwise re-raise
                error_msg = str(e).lower()
                if "already exists" not in error_msg and "conflict" not in error_msg:
                    print(f"Failed to create collection '{collection_name}': {e}")
                    raise e

        store = LambdaDBVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.get_embeddings(),
            validate_collection=bool(
                os.getenv("LAMBDADB_COLLECTION_NAME")
            ),  # Only validate if using existing collection
            default_consistent_read=True,  # Enable consistent reads for tests
        )

        # Ensure vectorstore starts empty for each test
        # Clear any existing documents in the collection
        try:
            # Use a broad query to find all documents
            all_docs = client.collections.query(
                collection_name=collection_name,
                query={"match_all": {}},
                size=100,  # Maximum allowed
                consistent_read=True,
            ).docs

            if all_docs:
                # Delete all found documents
                doc_ids = [doc.doc["id"] for doc in all_docs]
                client.collections.docs.delete(
                    collection_name=collection_name,
                    ids=doc_ids,
                )

        except Exception:
            # If clearing fails, continue - collection might be empty
            pass

        try:
            yield store
        finally:
            # Only cleanup if we created the collection (not using existing one)
            if not os.getenv("LAMBDADB_COLLECTION_NAME"):
                try:
                    client.collections.delete(collection_name=collection_name)
                except Exception:
                    # Cleanup failed - may need manual cleanup
                    pass
