from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbedding:

    def __init__(self, model_name):

        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, chunks):

        sentences = []

        for chunk in chunks:

            sentences.append(chunk.page_content)

        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        print("Embeddings Created Successfully")

        print("Embedding Type :", type(embeddings))

        print("Embedding Shape :", embeddings.shape)

        return embeddings