"""Weaviate RAG pipeline module."""

import json
from typing import Dict, List, Set

import weaviate
from weaviate.classes.config import Configure, DataType, Property
from weaviate.classes.query import Filter


class WeaviateRAGPipeline:
    """
    Store and retrieve document chunks in Weaviate with optional reranking.
    """

    def __init__(
        self,
        client: weaviate.WeaviateClient,
        embedding_model,
        reranker_model=None,
        collection_name: str = "PolicyDocuments",
        hybrid_search_limit: int = 10,
        hybrid_alpha: float = 0.65,
    ):
        """
        Initialize pipeline dependencies and collection settings.

        Args:
            client: Connected Weaviate client instance.
            embedding_model: Embedding model used to generate vectors.
            reranker_model: Optional reranker for post-retrieval sorting.
            collection_name: Target Weaviate collection name.
            hybrid_search_limit: Candidate count pulled before reranking.
            hybrid_alpha: Hybrid search alpha (keyword vs vector balance).
        """
        self.client = client
        self.embeddings = embedding_model
        self.reranker = reranker_model
        self.collection_name = collection_name
        self.hybrid_search_limit = hybrid_search_limit
        self.hybrid_alpha = hybrid_alpha

    def collection_exists(self) -> bool:
        """Check whether the configured collection exists in Weaviate."""
        return self.client.collections.exists(self.collection_name)

    def collection_has_data(self) -> bool:
        """Return True if the collection exists and contains at least one object."""
        if not self.collection_exists():
            return False
        collection = self.client.collections.use(self.collection_name)
        response = collection.aggregate.over_all(total_count=True)
        count = response.total_count or 0
        print(f"[INFO] Collection '{self.collection_name}' has {count} objects")
        return count > 0

    def delete_collection(self) -> None:
        """Delete the collection and all its data from Weaviate."""
        if self.collection_exists():
            self.client.collections.delete(self.collection_name)
            print(f"[STEP] Collection '{self.collection_name}' deleted")
        else:
            print(f"[INFO] Collection '{self.collection_name}' does not exist, nothing to delete")

    def get_object_count(self) -> int:
        """Return the total number of objects in the collection."""
        if not self.collection_exists():
            return 0
        collection = self.client.collections.use(self.collection_name)
        response = collection.aggregate.over_all(total_count=True)
        return response.total_count or 0

    def list_documents(self) -> List[str]:
        """Return a sorted list of unique file_name values stored in the collection."""
        if not self.collection_exists():
            return []
        collection = self.client.collections.use(self.collection_name)
        file_names: Set[str] = set()
        for item in collection.iterator(return_properties=["file_name"]):
            file_names.add(item.properties["file_name"])
        return sorted(file_names)

    def delete_document(self, file_name: str) -> int:
        """Delete all chunks belonging to a specific document and return count deleted."""
        if not self.collection_exists():
            return 0
        collection = self.client.collections.use(self.collection_name)
        result = collection.data.delete_many(
            where=Filter.by_property("file_name").equal(file_name)
        )
        deleted = result.successful
        print(f"[STEP] Deleted {deleted} chunks for document '{file_name}'")
        return deleted

    def create_collection(self) -> None:
        """
        Create collection schema if it does not already exist.
        """
        if self.collection_exists():
            print(f"[INFO] Collection '{self.collection_name}' already exists")
            return

        self.client.collections.create(
            name=self.collection_name,
            properties=[
                Property(name="text", data_type=DataType.TEXT),
                Property(name="title", data_type=DataType.TEXT),
                Property(name="heading", data_type=DataType.TEXT),
                Property(name="chunk_index", data_type=DataType.INT),
                Property(name="total_chunks", data_type=DataType.INT),
                Property(name="file_name", data_type=DataType.TEXT),
            ],
            vectorizer_config=Configure.Vectorizer.none(),
        )
        print(f"[STEP] Collection '{self.collection_name}' created")

    def store_chunks(self, final_chunks: List[Dict]) -> None:
        """
        Generate embeddings and store chunk objects in Weaviate.
        """
        collection = self.client.collections.use(self.collection_name)
        with collection.batch.dynamic() as batch:
            for chunk in final_chunks:
                vector = self.embeddings.embed_query(chunk["text"])
                properties = {
                    "text": chunk["text"],
                    "title": chunk["metadata"]["title"],
                    "heading": chunk["metadata"]["heading"],
                    "chunk_index": chunk["metadata"]["chunk_index"],
                    "total_chunks": chunk["metadata"]["total_chunks"],
                    "file_name": chunk["metadata"]["file_name"],
                }
                batch.add_object(properties=properties, vector=vector)

        failed_objects = getattr(collection.batch, "failed_objects", [])
        if failed_objects:
            print(f"[ERROR] Failed to insert {len(failed_objects)} objects")
        else:
            print("[INFO] No failed objects in Weaviate batch insert")
        print(f"[STEP] Stored {len(final_chunks)} chunks in Weaviate")

    def preview_stored_json(self, limit: int = 5) -> List[Dict]:
        """
        Fetch and print raw JSON-like objects currently stored in Weaviate.
        """
        collection = self.client.collections.use(self.collection_name)
        response = collection.query.fetch_objects(
            limit=limit,
            include_vector=True,
        )
        objects = self._format_results(response.objects)
        print(f"[DEBUG] Weaviate stored objects preview (limit={limit}):")
        print(json.dumps(objects, indent=2))
        return objects

    def query(self, user_query: str, limit: int = 3) -> List[Dict]:
        """
        Retrieve top hybrid results and optionally rerank with cross-encoder.
        """
        collection = self.client.collections.use(self.collection_name)
        query_vector = self.embeddings.embed_query(user_query)

        results = collection.query.hybrid(
            query=user_query,
            vector=query_vector,
            limit=self.hybrid_search_limit,
            alpha=self.hybrid_alpha,
            include_vector=True,
        )

        if self.reranker:
            passages = [obj.properties["text"] for obj in results.objects]
            scores = [
                self.reranker.similarity(
                    query_vector, self.embeddings.embed_query(passage)
                ).item()
                for passage in passages
            ]
            scored_results = sorted(
                zip(scores, results.objects), key=lambda item: item[0], reverse=True
            )
            final_objects = [obj for _score, obj in scored_results[:limit]]
        else:
            final_objects = results.objects[:limit]

        return self._format_results(final_objects)

    def _format_results(self, objects) -> List[Dict]:
        """Format Weaviate objects into a standard dict structure."""
        return [
            {
                "metadata": {
                    "text": obj.properties["text"],
                    "title": obj.properties["title"],
                    "heading": obj.properties["heading"],
                    "chunk_index": obj.properties["chunk_index"],
                    "total_chunks": obj.properties["total_chunks"],
                    "file_name": obj.properties["file_name"],
                },
                "embedding": (
                    obj.vector.get("default")
                    if isinstance(getattr(obj, "vector", None), dict)
                    else getattr(obj, "vector", None)
                ),
                "uuid": str(obj.uuid),
            }
            for obj in objects
        ]


def connect_local_weaviate(port: int = 8081, grpc_port: int = 50051) -> weaviate.WeaviateClient:
    """
    Connect to local Weaviate instance using Docker-exposed ports.
    """
    client = weaviate.connect_to_local(port=port, grpc_port=grpc_port)
    if client.is_ready():
        print("[STEP] Weaviate connected and ready")
    else:
        print("[ERROR] Weaviate connection failed")
    return client

