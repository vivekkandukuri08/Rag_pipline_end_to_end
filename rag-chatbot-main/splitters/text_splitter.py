from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextSplitter:

    def __init__(self, chunk_size=500, chunk_overlap=50):

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, documents):

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        chunks = splitter.split_documents(documents)

        print("Text Splitting Completed")
        print("Total Chunks :", len(chunks))

        return chunks