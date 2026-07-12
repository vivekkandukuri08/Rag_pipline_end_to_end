from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from langchain_core.documents import Document

from loaders.pdf_loader import PDFLoader
from splitters.text_splitter import TextSplitter
from embeddings.sentence_transformer import SentenceTransformerEmbedding
from vectorstores.pinecone_store import PineconeStore
from llm.gemini_llm import GeminiLLM

from config import (
    PDF_PATH,
    MODEL_NAME,
    PINECONE_API_KEY,
    INDEX_NAME,
    DIMENSION
)

# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# --------------------------------------------------
# LOAD PDF
# --------------------------------------------------

loader = PDFLoader(PDF_PATH)
documents = loader.load()

# --------------------------------------------------
# TEXT SPLITTING
# --------------------------------------------------

splitter = TextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split(documents)

# --------------------------------------------------
# CREATE EMBEDDING MODEL
# --------------------------------------------------

embedding_model = SentenceTransformerEmbedding(MODEL_NAME)

# --------------------------------------------------
# CREATE DOCUMENT EMBEDDINGS
# --------------------------------------------------

embeddings = embedding_model.create_embeddings(chunks)

# --------------------------------------------------
# CONNECT PINECONE
# --------------------------------------------------

pinecone_store = PineconeStore()

# --------------------------------------------------
# SAVE VECTORS
# --------------------------------------------------

pinecone_store.save_vectors(
    embeddings,
    chunks
)

# --------------------------------------------------
# CREATE GEMINI LLM
# --------------------------------------------------

llm = GeminiLLM()

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )

# --------------------------------------------------
# ASK QUESTION
# --------------------------------------------------

@app.post("/", response_class=HTMLResponse)
async def ask_question(request: Request):

    form = await request.form()

    question = form["question"]

    print("\n======================================")
    print("User Question:", question)
    print("======================================")

    # --------------------------------------------------
    # CREATE QUESTION DOCUMENT
    # --------------------------------------------------

    question_document = Document(
        page_content=question
    )

    # --------------------------------------------------
    # CREATE QUESTION EMBEDDING
    # --------------------------------------------------

    question_embedding = embedding_model.create_embeddings(
        [question_document]
    )[0]

    print("Question Embedding Created")

    # --------------------------------------------------
    # SEARCH PINECONE
    # --------------------------------------------------

    search_result = pinecone_store.search(
        question_embedding,
        top_k=10
    )

    # --------------------------------------------------
    # RETRIEVE CONTEXT
    # --------------------------------------------------

    context_list = []

    print("\n========== RETRIEVED DOCUMENTS ==========")

    for match in search_result.matches:

        print("\n--------------------------------")

        print("Score:", match.score)

        text = None

        if match.metadata:
            text = match.metadata.get("text")

        print("\nRetrieved Text:")
        print(text)

        if text:
            context_list.append(text)

    # --------------------------------------------------
    # COMBINE CONTEXT
    # --------------------------------------------------

    context = "\n\n".join(context_list)

    print("\n======================================")
    print("FINAL CONTEXT")
    print("======================================")
    print(context)

    # --------------------------------------------------
    # CHECK CONTEXT
    # --------------------------------------------------

    if not context.strip():

        answer = "Answer not found in the document."

    else:

        # --------------------------------------------------
        # GENERATE ANSWER USING GEMINI
        # --------------------------------------------------

        answer = llm.generate_answer(
            question,
            context
        )

    # --------------------------------------------------
    # PRINT FINAL ANSWER
    # --------------------------------------------------

    print("\n======================================")
    print("FINAL ANSWER:")
    print(answer)
    print("======================================")

    # --------------------------------------------------
    # RETURN ANSWER TO HTML
    # --------------------------------------------------

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "answer": answer
        }
    )