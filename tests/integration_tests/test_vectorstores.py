import logging
import os
import time
import uuid
from typing import Generator

import pytest
from dotenv import load_dotenv
from lambdadb import LambdaDB
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_lambdadb.vectorstores import LambdaDBVectorStore

load_dotenv()


class TestLambdaDBVectorStore(VectorStoreIntegrationTests):
    """Integration tests for LambdaDB vector store.

    To run with real LambdaDB service, set these environment variables:
    - LAMBDADB_BASE_URL: API base URL (default: https://api.lambdadb.ai)
    - LAMBDADB_PROJECT_NAME: Project name (default: playground)
    - LAMBDADB_PROJECT_API_KEY: Your LambdaDB API key

    Debugging: set LAMBDADB_KEEP_COLLECTION_ON_FAILURE=1 to leave failed-test
    collections intact (no cleanup) so you can inspect them.
    """

    @pytest.fixture()
    def vectorstore(
        self, request: pytest.FixtureRequest
    ) -> Generator[VectorStore, None, None]:  # type: ignore
        """Get an empty vectorstore for unit tests."""
        # Initialize LambdaDB client (0.7.0+ base_url + project_name)
        client = LambdaDB(
            base_url=os.getenv("LAMBDADB_BASE_URL", "https://api.lambdadb.ai"),
            project_name=os.getenv("LAMBDADB_PROJECT_NAME", "playground"),
            project_api_key=os.getenv("LAMBDADB_PROJECT_API_KEY"),
        )

        # Use env var for collection name, or generate unique one for testing
        if os.getenv("LAMBDADB_COLLECTION_NAME"):
            collection_name = os.getenv("LAMBDADB_COLLECTION_NAME")
        else:
            # Unique collection name (max 52 chars): short UUID + timestamp
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

                logging.info(
                    "Creating collection '%s' with dimension %s",
                    collection_name,
                    dimension,
                )
                client.collections.create(
                    collection_name=collection_name,
                    index_configs={
                        "vector": {
                            "type": "vector",
                            "dimensions": dimension,
                            "similarity": "cosine",
                        },
                        "page_content": {
                            "type": "text",
                            "analyzers": ["english"],
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
                            collection_name=collection_name,
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
                    logging.error(
                        "Failed to create collection '%s': %s",
                        collection_name,
                        e,
                    )
                    raise e

        store = LambdaDBVectorStore(
            client=client,
            collection_name=collection_name,
            text_field="page_content",
            vector_field="vector",
            embedding=self.get_embeddings(),
            validate_collection=bool(
                os.getenv("LAMBDADB_COLLECTION_NAME")
            ),  # Only validate if using existing collection
            default_consistent_read=True,  # Enable consistent reads for tests
        )

        # Ensure vectorstore starts empty for each test (delete by query *:*)
        try:
            coll = client.collection(collection_name)
            coll.docs.delete(query_filter={"queryString": {"query": "*:*"}})
        except Exception:
            pass

        try:
            yield store
        finally:
            # Only cleanup if we created the collection (not using existing one)
            if not os.getenv("LAMBDADB_COLLECTION_NAME"):
                # Skip cleanup on failure when LAMBDADB_KEEP_COLLECTION_ON_FAILURE=1
                keep_on_failure = os.getenv(
                    "LAMBDADB_KEEP_COLLECTION_ON_FAILURE", ""
                ).lower() in ("1", "true", "yes")
                if keep_on_failure and hasattr(request.node, "rep_call"):
                    rep = getattr(request.node, "rep_call", None)
                    if rep is not None and getattr(rep, "failed", False):
                        return  # Leave collection for debugging
                try:
                    client.collections.delete(collection_name=collection_name)
                except Exception:
                    # Cleanup failed - may need manual cleanup
                    pass
