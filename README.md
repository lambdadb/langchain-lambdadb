# langchain-lambdadb

This package contains the LangChain integration with LambdaDB

## Installation

```bash
pip install -U langchain-lambdadb
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatLambdaDB` class exposes chat models from LambdaDB.

```python
from langchain_lambdadb import ChatLambdaDB

llm = ChatLambdaDB()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`LambdaDBEmbeddings` class exposes embeddings from LambdaDB.

```python
from langchain_lambdadb import LambdaDBEmbeddings

embeddings = LambdaDBEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`LambdaDBLLM` class exposes LLMs from LambdaDB.

```python
from langchain_lambdadb import LambdaDBLLM

llm = LambdaDBLLM()
llm.invoke("The meaning of life is")
```
