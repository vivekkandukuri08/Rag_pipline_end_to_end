from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter
from embeddings.sentence_transformer import SentenceTransformerEmbedding
from vectorstores.pinecone_store import PineconeStore

from config import (
    PDF_PATH,
    MODEL_NAME
)


print("RAG INDEXING STARTED")


# --------------------------------
# LOAD PDF
# --------------------------------

loader = PDFLoader(
    PDF_PATH
)

documents = loader.load()


# --------------------------------
# SPLIT DOCUMENT
# --------------------------------

splitter = TextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split(
    documents
)


# --------------------------------
# CREATE EMBEDDINGS
# --------------------------------

embedding_model = SentenceTransformerEmbedding(
    MODEL_NAME
)

embeddings = embedding_model.create_embeddings(
    chunks
)


# --------------------------------
# CONNECT PINECONE
# --------------------------------

pinecone_store = PineconeStore()


# --------------------------------
# DELETE OLD VECTORS
# --------------------------------

pinecone_store.delete_vectors()


# --------------------------------
# SAVE NEW VECTORS
# --------------------------------

pinecone_store.save_vectors(
    embeddings,
    chunks
)


print("RAG INDEXING COMPLETED")