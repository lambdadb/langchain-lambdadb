"""LambdaDB vector stores."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import (
    Any,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np

from lambdadb import LambdaDB
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import maximal_marginal_relevance

VST = TypeVar("VST", bound=VectorStore)


class LambdaDBVectorStore(VectorStore):
    """LambdaDB vector store integration.

    This integration works with existing LambdaDB collections. The collection must be
    created beforehand with proper vector and text indexes configured.

    Setup:
        Install ``langchain-lambdadb`` package.

        .. code-block:: bash

            pip install -U langchain-lambdadb

    Key init args — indexing params:
        collection_name: str
            Name of an existing collection in LambdaDB.
        embedding: Embeddings
            Embedding function to use.

    Key init args — client params:
        client: LambdaDB
            LambdaDB client instance.

    Instantiate:
        .. code-block:: python

            from langchain_lambdadb.vectorstores import LambdaDBVectorStore
            from langchain_openai import OpenAIEmbeddings
            from lambdadb import LambdaDB

            # Initialize client
            client = LambdaDB(
                project_url="<your_project_url>",
                project_api_key="<your_project_api_key>"
            )

            # Use existing collection
            vector_store = LambdaDBVectorStore(
                collection_name="my_existing_collection",
                embedding=OpenAIEmbeddings(),
                client=client,
            )

    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'baz': 'bar'}]

    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"queryString":{"query":"baz:bar"}})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'baz': 'bar'}]

    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.000000] qux [{'bar': 'baz', 'baz': 'bar'}]

    Async:
        .. code-block:: python

            # add documents
            await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            await vector_store.adelete(ids=["3"])

            # search
            results = await vector_store.asimilarity_search(query="thud", k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux", k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * [SIM=0.892341] qux [{'bar': 'baz', 'baz': 'bar'}]

    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 2, "score_threshold": 0.5},
            )
            relevant_docs = retriever.invoke("thud")
            for doc in relevant_docs:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            * thud [{'bar': 'baz'}]
            * foo [{'baz': 'bar'}]

    """  # noqa: E501

    def __init__(
        self,
        client: LambdaDB,
        collection_name: str,
        embedding: Embeddings,
        text_field: str = "text",
        vector_field: str = "vector",
        validate_collection: bool = True,
        default_consistent_read: bool = False,
    ) -> None:
        """Initialize with the given embedding function.

        Args:
            client: LambdaDB client. Documentation: https://docs.lambdadb.ai
            collection_name: Name of an existing collection in LambdaDB.
                The collection must already exist and have proper vector indexes
                configured.
            embedding: embedding function to use.
            text_field: Name of the text field in documents (default: "text").
            vector_field: Name of the vector field in documents (default: "vector").
            validate_collection: Whether to validate that the collection exists and is
                active (default: True).
            default_consistent_read: Default value for consistent_read parameter in all
                read operations. When True, ensures immediate consistency but may have
                slight performance impact. When False, uses eventual consistency which
                is faster but may return stale data for ~1 minute after writes
                (default: False).

        Note:
            This integration is designed to work with existing LambdaDB collections.
            The collection should be created beforehand with appropriate vector and text
            indexes.
        """
        if client is None or not isinstance(client, LambdaDB):
            raise ValueError(
                f"client value can't be None "
                f"and should be an instance of lambdadb.LambdaDB, "
                f"got {type(client)}"
            )

        if embedding is None:
            raise ValueError(
                "`embedding` value can't be None. Pass `Embeddings` instance."
            )

        if not collection_name or not isinstance(collection_name, str):
            raise ValueError(
                f"collection_name must be a non-empty string, got {collection_name}"
            )

        # Validate that the collection exists and is active
        if validate_collection:
            try:
                collection_info = client.collections.get(
                    collection_name=collection_name
                )
                if collection_info.collection.collection_status.value != "ACTIVE":
                    raise ValueError(
                        f"Collection '{collection_name}' exists but is not ACTIVE. "
                        f"Status: {collection_info.collection.collection_status.value}"
                    )
            except Exception as e:
                raise ValueError(
                    f"Collection '{collection_name}' does not exist or is not "
                    f"accessible. "
                    f"Please create the collection first with proper vector and text "
                    f"indexes. "
                    f"Error: {e}"
                )

        self._client = client
        self._collection_name = collection_name
        self.embedding = embedding
        self._text_field = text_field
        self._vector_field = vector_field
        self._default_consistent_read = default_consistent_read

    @classmethod
    def from_texts(  # type: ignore[override]
        cls: Type[LambdaDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        *,
        client: LambdaDB,
        collection_name: str,
        ids: Optional[List[str]] = None,
        validate_collection: bool = True,
        default_consistent_read: bool = True,
        **kwargs: Any,
    ) -> LambdaDBVectorStore:
        """Create a LambdaDBVectorStore from a list of texts.

        Args:
            texts: List of texts to add to the vectorstore.
            embedding: Embedding function to use.
            metadatas: Optional list of metadata dicts for each text.
            client: LambdaDB client instance.
            collection_name: Name of existing collection to use.
            ids: Optional list of IDs for the texts.
            validate_collection: Whether to validate collection exists and is active.
            default_consistent_read: Default value for consistent_read parameter.
            **kwargs: Additional arguments passed to add_texts.

        Returns:
            LambdaDBVectorStore instance with the texts added.
        """
        store = cls(
            client=client,
            collection_name=collection_name,
            embedding=embedding,
            validate_collection=validate_collection,
            default_consistent_read=default_consistent_read,
            **{k: v for k, v in kwargs.items() if k in ["text_field", "vector_field"]},
        )
        store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
        return store

    @property
    def embeddings(self) -> Embeddings:
        return self.embedding

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Run more texts through the embeddings and add to the vectorstore.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            ids:
                Optional list of ids to associate with the texts. Ids have to be
                uuid-like strings.

        Returns:
            List of ids from adding the texts into the vectorstore.
        """
        texts = list(texts)

        for index, text in enumerate(texts):
            # 50KB is the max size of a document for LambdaDB
            # Measuring only on the text part is technically insufficient, but it still provides a good guide.
            if 50 * 1000 < len(text):
                raise ValueError(
                    f"The text at index {index} is too long. Max length is 50KB."
                )

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            if len(ids) != len(texts):
                msg = (
                    f"ids must be the same length as texts. "
                    f"Got {len(ids)} ids and {len(texts)} texts."
                )
                raise ValueError(msg)

            ids = [str(id) if id is not None else str(uuid.uuid4()) for id in ids]

        vectors = self.embedding.embed_documents(texts)

        # Prepare all documents for bulk upsert
        docs = []
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if metadatas else {}
            docs.append(
                {
                    "id": ids[idx],
                    self._text_field: text,
                    self._vector_field: vectors[idx],
                    "metadata": metadata,
                }
            )

        # Use regular upsert method for consistent immediate indexing
        # Process in batches to stay under 6MB limit per request
        # SAFETY: Setting batch size to 100 is safe, because we've checked that there is no document longer than 50KB.
        added_ids = []
        batch_size = 100  # Conservative batch size for 6MB limit

        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]

            try:
                self._client.collections.docs.upsert(
                    collection_name=self._collection_name, docs=batch_docs
                )
                added_ids.extend(batch_ids)
            except Exception as e:
                raise RuntimeError(f"Upsert operation failed: {str(e)}") from e

        return added_ids

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        # Extract IDs from documents if they have them, or use provided ids
        ids = kwargs.get("ids")
        if ids is None:
            # Try to get IDs from the documents themselves
            ids = [doc.id if doc.id is not None else None for doc in documents]
            # If all are None, let add_texts handle ID generation
            if all(id is None for id in ids):
                ids = None
        # Remove ids from kwargs to avoid duplicate parameter
        filtered_kwargs = {k: v for k, v in kwargs.items() if k != "ids"}
        return self.add_texts(texts, metadatas, ids=ids, **filtered_kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        if ids:
            try:
                self._client.collections.docs.delete(
                    collection_name=self._collection_name,
                    ids=ids,
                )
            except Exception as e:
                # Handle cases where documents don't exist gracefully
                error_message = str(e).lower()
                if (
                    "not found" in error_message
                    or "does not exist" in error_message
                    or "creating state" in error_message
                    or "badrequest" in error_message
                ):
                    # Silently ignore missing documents or temporary state issues
                    pass
                else:
                    raise (e)

    def _build_langchain_document(self, doc: dict) -> Document:
        return Document(
            id=doc["id"],
            page_content=doc[self._text_field],
            metadata=doc["metadata"],
        )

    def get_by_ids(
        self, ids: Sequence[str], /, consistent_read: Optional[bool] = None
    ) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.
            consistent_read: Whether to use consistent read. If None, uses the default
                setting from initialization.

        Returns:
            A list of Document objects in the same order as input IDs.
        """
        if consistent_read is None:
            consistent_read = self._default_consistent_read

        fetched_docs = self._client.collections.docs.fetch(
            collection_name=self._collection_name,
            ids=list(ids),
            consistent_read=consistent_read,
        ).docs

        # Create a mapping from ID to Document since LambdaDB doesn't preserve order
        doc_map = {}
        for doc in fetched_docs:
            langchain_doc = self._build_langchain_document(doc.doc)
            doc_map[langchain_doc.id] = langchain_doc

        # Return documents in the same order as requested IDs
        # Skip IDs that weren't found (LangChain tests expect this behavior)
        ordered_docs = []
        for id in ids:
            if id in doc_map:
                ordered_docs.append(doc_map[id])

        return ordered_docs

    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
    ) -> List[tuple[Document, float]]:
        # Use proper LambdaDB knn query format
        query = {"knn": {"field": self._vector_field, "queryVector": embedding, "k": k}}

        # If filter provided, add it to the knn query
        if filter:
            query["knn"]["filter"] = filter

        if consistent_read is None:
            consistent_read = self._default_consistent_read

        docs = self._client.collections.query(
            collection_name=self._collection_name,
            size=k,
            query=query,
            consistent_read=consistent_read,
        ).docs

        if not docs:
            return []

        results = []
        for doc in docs:
            langchain_doc = self._build_langchain_document(doc.doc)
            # LambdaDB returns relevance score in doc.score
            score: float = (
                float(doc.score)
                if hasattr(doc, "score") and doc.score is not None
                else 1.0
            )
            results.append((langchain_doc, score))

        return results

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return [
            doc
            for doc, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding,
                k=k,
                filter=filter,
                consistent_read=consistent_read,
                **kwargs,
            )
        ]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return self._similarity_search_with_score_by_vector(
            embedding=embedding,
            k=k,
            filter=filter,
            consistent_read=consistent_read,
            **kwargs,
        )

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [
            doc
            for doc, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, **kwargs
            )
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to None.
            consistent_read: Whether to use consistent read. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.embedding.embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            consistent_read=consistent_read,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch to pass to MMR algorithm.
                     Defaults to 20.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
            filter: Filter by metadata. Defaults to None.
            consistent_read: Whether to use consistent read. Defaults to None.

        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        # Use proper LambdaDB knn query format to fetch more documents
        query = {
            "knn": {"field": self._vector_field, "queryVector": embedding, "k": fetch_k}
        }

        # If filter provided, add it to the knn query
        if filter:
            query["knn"]["filter"] = filter

        if consistent_read is None:
            consistent_read = self._default_consistent_read

        # Query with include_vectors to get the embeddings for MMR
        try:
            docs = self._client.collections.query(
                collection_name=self._collection_name,
                size=fetch_k,
                query=query,
                consistent_read=consistent_read,
                include_vectors=True,  # Essential for MMR
            ).docs
        except TypeError:
            # Fallback if include_vectors parameter is not supported
            docs = self._client.collections.query(
                collection_name=self._collection_name,
                size=fetch_k,
                query=query,
                consistent_read=consistent_read,
            ).docs

        if not docs:
            return []

        # Extract embeddings from the documents
        embeddings_list = []
        documents_list = []

        for doc in docs:
            langchain_doc = self._build_langchain_document(doc.doc)
            documents_list.append(langchain_doc)

            # Try to get vector from document
            doc_vector = None
            if hasattr(doc, "doc") and hasattr(doc.doc, self._vector_field):
                doc_vector = getattr(doc.doc, self._vector_field)
            elif isinstance(doc.doc, dict) and self._vector_field in doc.doc:
                doc_vector = doc.doc[self._vector_field]

            if doc_vector is not None:
                embeddings_list.append(doc_vector)
            else:
                # If vector is not included, we can't do MMR - fallback to regular search
                return documents_list[:k]

        # Apply MMR algorithm (convert to numpy arrays)
        mmr_indexes = maximal_marginal_relevance(
            query_embedding=np.array(embedding),
            embedding_list=embeddings_list,  # List of vectors
            lambda_mult=lambda_mult,
            k=k,
        )

        # Return documents selected by MMR
        return [documents_list[i] for i in mmr_indexes]

    ### ASYNC METHODS ###

    async def aadd_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[list[dict]] = None,
        ids: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async version of add_texts.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        # LambdaDB client doesn't have async support yet, so we run sync version
        return self.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)

    async def aadd_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Async version of add_documents.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.add_documents(documents=documents, **kwargs)

    async def adelete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        """Async version of delete.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.delete(ids=ids, **kwargs)

    async def aget_by_ids(
        self, ids: Sequence[str], /, consistent_read: Optional[bool] = None
    ) -> list[Document]:
        """Async version of get_by_ids.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.get_by_ids(ids, consistent_read=consistent_read)

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of similarity_search.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.similarity_search(
            query=query,
            k=k,
            filter=filter,
            consistent_read=consistent_read,
            **kwargs,
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Async version of similarity_search_with_score.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter,
            consistent_read=consistent_read,
            **kwargs,
        )

    async def asimilarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Async version of similarity_search_by_vector.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.similarity_search_by_vector(embedding=embedding, k=k, **kwargs)

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of max_marginal_relevance_search.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.max_marginal_relevance_search(
            query=query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            consistent_read=consistent_read,
            **kwargs,
        )

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: Optional[bool] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Async version of max_marginal_relevance_search_by_vector.

        Note: Currently runs synchronously as LambdaDB client doesn't support async
        operations.
        """
        return self.max_marginal_relevance_search_by_vector(
            embedding=embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            consistent_read=consistent_read,
            **kwargs,
        )
