"""LambdaDB vector stores."""

from __future__ import annotations

from collections.abc import Iterable
import uuid
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

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.utils import _cosine_similarity as cosine_similarity
from lambdadb import LambdaDB

VST = TypeVar("VST", bound=VectorStore)


class LambdaDBVectorStore(VectorStore):

    """LambdaDB vector store integration.

    Setup:
        Install ``langchain-lambdadb`` package.

        .. code-block:: bash

            pip install -U langchain-lambdadb

    Key init args — indexing params:
        collection_name: str
            Name of the collection.
        embedding_function: Embeddings
            Embedding function to use.

    Key init args — client params:
        client: Optional[Client]
            Client to use.
        connection_args: Optional[dict]
            Connection arguments.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_lambdadb.vectorstores import LambdaDBVectorStore
            from langchain_openai import OpenAIEmbeddings

            vector_store = LambdaDBVectorStore(
                collection_name="foo",
                embedding_function=OpenAIEmbeddings(),
                connection_args={"uri": "./foo.db"},
                # other params...
            )

    # TODO: Populate with relevant variables.
    Add Documents:
        .. code-block:: python

            from langchain_core.documents import Document

            document_1 = Document(page_content="foo", metadata={"baz": "bar"})
            document_2 = Document(page_content="thud", metadata={"bar": "baz"})
            document_3 = Document(page_content="i will be deleted :(")

            documents = [document_1, document_2, document_3]
            ids = ["1", "2", "3"]
            vector_store.add_documents(documents=documents, ids=ids)

    # TODO: Populate with relevant variables.
    Delete Documents:
        .. code-block:: python

            vector_store.delete(ids=["3"])

    # TODO: Fill out with relevant variables and example output.
    Search:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1)
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Search with filter:
        .. code-block:: python

            results = vector_store.similarity_search(query="thud",k=1,filter={"bar": "baz"})
            for doc in results:
                print(f"* {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

    # TODO: Fill out with relevant variables and example output.
    Search with score:
        .. code-block:: python

            results = vector_store.similarity_search_with_score(query="qux",k=1)
            for doc, score in results:
                print(f"* [SIM={score:3f}] {doc.page_content} [{doc.metadata}]")

        .. code-block:: python

            # TODO: Example output

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
            collection_name: str,
            embedding: Embeddings) -> None:
        """Initialize with the given embedding function.

        Args:
            client: LambdaDB client. Documentation: https://docs.lambdadb.ai
            collection_name: Collection name.
            embedding: embedding function to use.
        """
        if not isinstance(client, LambdaDB):
            raise ValueError(
                f"client should be an instance of lambdadb.LambdaDB, "
                f"got {type(client)}"
            )
        
        if embedding is None:
            raise ValueError(
                "`embedding` value can't be None. Pass `Embeddings` instance."
            )
        

        self._database: dict[str, dict[str, Any]] = {}
        self._client = client
        self._collection_name = collection_name
        self.embedding = embedding

    @classmethod
    def from_texts(
        cls: Type[LambdaDBVectorStore],
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> LambdaDBVectorStore:
        store = cls(
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
        
        if metadatas and len(metadatas) != len(texts):
            msg = (
                    f"metadatas must be the same length as texts. "
                    f"Got {len(metadatas)} metadatas and {len(texts)} texts."
                )
            raise ValueError(msg)

        vectors = self.embedding.embed_documents(texts)

        added_ids = []
        batch_size = 100
        docs_count = 0
        docs = []
        doc_ids = []
        
        for idx, text in enumerate(texts):
            doc_id = ids[idx]
            doc_ids.append(doc_id)
            vector = vectors[idx]
            doc = {
                "id": doc_id,
                "vector": vector,
            }
            docs.append(doc)
            docs_count += 1

            if docs_count == batch_size:
                res = self._client.collections.docs.upsert(
                    project_name="",
                    collection_name=self._collection_name,
                    docs=docs
                )
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
            for _id in ids:
                self._database.pop(_id, None)

    def get_by_ids(self, ids: Sequence[str], /) -> list[Document]:
        """Get documents by their ids.

        Args:
            ids: The ids of the documents to get.

        Returns:
            A list of Document objects.
        """
        documents = []

        for doc_id in ids:
            doc = self._database.get(doc_id)
            if doc:
                documents.append(
                    Document(
                        id=doc["id"],
                        page_content=doc["text"],
                        metadata=doc["metadata"],
                    )
                )
        return documents

    # NOTE: the below helper method implements similarity search for in-memory
    # storage. It is optional and not a part of the vector store interface.
    def _similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Callable[[Document], bool]] = None,
        **kwargs: Any,
    ) -> List[tuple[Document, float, List[float]]]:
        # get all docs with fixed order in list
        docs = list(self._database.values())

        if filter is not None:
            docs = [
                doc
                for doc in docs
                if filter(Document(page_content=doc["text"], metadata=doc["metadata"]))
            ]

        if not docs:
            return []

        similarity = cosine_similarity([embedding], [doc["vector"] for doc in docs])[0]

        # get the indices ordered by similarity score
        top_k_idx = similarity.argsort()[::-1][:k]

        return [
            (
                # Document
                Document(
                    id=doc_dict["id"],
                    page_content=doc_dict["text"],
                    metadata=doc_dict["metadata"],
                ),
                # Score
                float(similarity[idx].item()),
                # Embedding vector
                doc_dict["vector"],
            )
            for idx in top_k_idx
            # Assign using walrus operator to avoid multiple lookups
            if (doc_dict := docs[idx])
        ]

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        embedding = self.embedding.embed_query(query)
        return [
            doc
            for doc, _, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, **kwargs
            )
        ]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        embedding = self.embedding.embed_query(query)
        return [
            (doc, similarity)
            for doc, similarity, _ in self._similarity_search_with_score_by_vector(
                embedding=embedding, k=k, **kwargs
            )
        ]

    ### ADDITIONAL OPTIONAL SEARCH METHODS BELOW ###

    # def similarity_search_by_vector(
    #     self, embedding: List[float], k: int = 4, **kwargs: Any
    # ) -> List[Document]:
    #     raise NotImplementedError

    # def max_marginal_relevance_search(
    #     self,
    #     query: str,
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError

    # def max_marginal_relevance_search_by_vector(
    #     self,
    #     embedding: List[float],
    #     k: int = 4,
    #     fetch_k: int = 20,
    #     lambda_mult: float = 0.5,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     raise NotImplementedError

