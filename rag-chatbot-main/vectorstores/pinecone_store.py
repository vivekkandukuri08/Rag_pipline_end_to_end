from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    INDEX_NAME
)


class PineconeStore:

    def __init__(self):

        self.pc = Pinecone(
            api_key=PINECONE_API_KEY
        )

        self.index = self.pc.Index(
            INDEX_NAME
        )

        print("Pinecone Index Connected")


    def save_vectors(self, embeddings, chunks):

        vectors = []

        for i, embedding in enumerate(embeddings):

            vectors.append(
                {
                    "id": f"id-{i}",
                    "values": embedding.tolist(),
                    "metadata": {
                        "text": chunks[i].page_content
                    }
                }
            )

        self.index.upsert(
            vectors=vectors
        )

        print("Vectors Saved Successfully")


    def search(self, query_embedding, top_k=10):

        result = self.index.query(
            vector=query_embedding.tolist(),
            top_k=top_k,
            include_metadata=True
        )

        return result