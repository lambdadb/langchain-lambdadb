"""LambdaDB vector stores."""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from typing import (
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

from lambdadb import LambdaDB
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import _cosine_similarity as cosine_similarity

VST = TypeVar("VST", bound=VectorStore)


class LambdaDBVectorStore(VectorStore):

    """LambdaDB vector store integration.

    Setup:
        Install ``langchain-lambdadb`` package.

        .. code-block:: bash

            pip install -U langchain-lambdadb

    Key init args — indexing params:
        project_name: str
            Name of the project.
        collection_name: str
            Name of the collection.
        embedding_function: Embeddings
            Embedding function to use.

    Key init args — client params:
        client: LambdaDB
            Client to use.
        base_url: str
            Base URL of the LambdaDB endpoint.

    Instantiate:
        .. code-block:: python

            from langchain_lambdadb.vectorstores import LambdaDBVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = LambdaDBVectorStore(
                project_name="foo"
                collection_name="bar",
                embedding_function=OpenAIEmbeddings(),
                client=LambdaDB(project_api_key=<project_api_key>),
                base_url="https://api.lambdadb.ai"
                # other params...
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

    # TODO: Fill out with relevant variables and example output.
    Async:
        .. code-block:: python

            # add documents
            # await vector_store.aadd_documents(documents=documents, ids=ids)

            # delete documents
            # await vector_store.adelete(ids=["3"])

            # search
            # results = vector_store.asimilarity_search(query="thud",k=1)

            # search with score
            results = await vector_store.asimilarity_search_with_score(query="qux",k=1)
            for doc,score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Use as Retriever:
        .. code-block:: python

            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
            )
            retriever.invoke("thud")

        .. code-block:: python

            # TODO: Example output

    """  # noqa: E501

    def __init__(
            self,
            client: LambdaDB,
            base_url: str,
            project_name: str,
            collection_name: str,
            embedding: Embeddings,
            text_field: str = "text",
            vector_field: str = "vector") -> None:
        """Initialize with the given embedding function.

        Args:
            client: LambdaDB client. Documentation: https://docs.lambdadb.ai
            collection_name: Collection name.
            embedding: embedding function to use.
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
        
        self._client = client
        self._base_url = base_url
        self._project_name = project_name
        self._collection_name = collection_name
        self.embedding = embedding
        self._text_field = text_field
        self._vector_field = vector_field

    @classmethod
    def from_texts(
        cls: Type[LambdaDBVectorStore],
        texts: List[str],
        client: LambdaDB,
        base_url: str,
        project_name: str,
        collection_name: str,
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> LambdaDBVectorStore:
        store = cls(
            client=client,
            base_url=base_url,
            project_name=project_name,
            collection_name=collection_name,
            embedding=embedding,
        )
        store.add_texts(texts=texts, metadatas=metadatas, **kwargs)
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
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        else:
            if len(ids) != len(texts):
                msg = (
                    f"ids must be the same length as texts. "
                    f"Got {len(ids)} ids and {len(texts)} texts."
                )
                raise ValueError(msg)
        
            ids = [id if id is not None else str(uuid.uuid4()) for id in ids]

        vectors = self.embedding.embed_documents(texts)

        added_ids = []
        batch_size = 100
        docs_count = 0
        docs = []
        doc_ids = []
        
        for idx, text in enumerate(texts):
            metadata = metadatas[idx] if metadatas else {}
            doc_ids.append(ids[idx])
            docs.append(
                {
                    "id": ids[idx],
                    self._text_field: text,
                    self._vector_field: vectors[idx],
                    "metadata": metadata,
                }
            )
            docs_count += 1

            if docs_count == batch_size:
                try:
                    self._client.collections.docs.upsert(
                        project_name=self._project_name,
                        collection_name=self._collection_name,
                        docs=docs,
                        server_url=self._base_url
                    )
                except Exception as e:
                    raise(e)

                added_ids.extend(doc_ids)
                docs_count = 0
                doc_ids.clear()

        return added_ids

    def add_documents(
        self,
        documents: List[Document],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the store."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.add_texts(texts, metadatas, **kwargs)

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> None:
        if ids:
            try:
                self._client.collections.docs.delete(
                    project_name=self._project_name,
                    collection_name=self._collection_name,
                    ids=ids,
                    server_url=self._base_url
                )
            except Exception as e:
                raise(e)

    def _build_langchain_document(self, doc: dict) -> Document:
        return Document(
            id=doc["id"],
            page_content=doc[self._text_field],
            metadata=doc["metadata"],
        )

    def get_by_ids(self, ids: Sequence[str], /, consistent_read: bool = False) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        langchain_docs = []

        fetched_docs = self._client.collections.docs.fetch(
            project_name=self._project_name,
            collection_name=self._collection_name,
            ids=list(ids),
            consistent_read=consistent_read,
            server_url=self._base_url
        ).docs

        for doc in fetched_docs:
            langchain_docs.append(
                self._build_langchain_document(doc.doc)
            )

        return langchain_docs

    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[dict[str, Any]] = None,
        consistent_read: bool = False,
        **kwargs: Any,
    ) -> List[tuple[Document, float, List[float]]]:        

        query = {
            "query": {
                self._vector_field: embedding,
                "k": k,
                "filter": filter
            }
        }

        docs = self._client.collections.query(
            project_name=self._project_name,
            collection_name=self._collection_name,
            size=k,
            query=query,
            consistent_read=consistent_read,
            server_url=self._base_url
        ).docs

        if not docs:
            return []

        langchain_docs = []
        for doc in docs:
            langchain_docs.append(
                self._build_langchain_document(doc.doc)
            )

        return langchain_docs

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[dict[str, Any]] = None,
        consistent_read: bool = False, **kwargs: Any
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, consistent_read=consistent_read, **kwargs
            )
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[dict[str, Any]] = None,
        consistent_read: bool = False, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, consistent_read=consistent_read, **kwargs
            )
        ]

    ### ADDITIONAL OPTIONAL SEARCH METHODS BELOW ###

    def similarity_search_by_vector(
        self, embedding: List[float], k: int = 4, **kwargs: Any
    ) -> List[Document]:
        raise NotImplementedError

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> List[Document]:
        raise NotImplementedError

