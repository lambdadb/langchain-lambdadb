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
    def vectorstore(  # type: ignore[override]
        self, request: pytest.FixtureRequest
    ) -> Generator[VectorStore, None, None]:
        """Get an empty vectorstore for unit tests."""
        # Initialize LambdaDB client (0.7.0+ base_url + project_name)
        client = LambdaDB(
            base_url=os.getenv("LAMBDADB_BASE_URL", "https://api.lambdadb.ai"),
            project_name=os.getenv("LAMBDADB_PROJECT_NAME", "playground"),
            project_api_key=os.getenv("LAMBDADB_PROJECT_API_KEY"),
        )

        # Use env var for collection name, or generate unique one for testing
        env_collection = os.getenv("LAMBDADB_COLLECTION_NAME")
        if env_collection:
            collection_name: str = env_collection
            use_existing = True
        else:
            timestamp = int(time.time()) % 1000000
            short_uuid = uuid.uuid4().hex[:8]
            collection_name = f"test_{short_uuid}_{timestamp}"
            use_existing = False

        # VectorStore creates collection if missing (create_if_not_exists=True)
        store = LambdaDBVectorStore(
            client=client,
            collection_name=collection_name,
            text_field="page_content",
            vector_field="vector",
            embedding=self.get_embeddings(),
            validate_collection=use_existing,
            default_consistent_read=True,
            create_if_not_exists=True,
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
            if not use_existing:
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
