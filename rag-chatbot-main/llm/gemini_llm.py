import os

from dotenv import load_dotenv
from google import genai

load_dotenv()


class GeminiLLM:

    def __init__(self):

        self.api_key = os.getenv("GEMINI_API_KEY")

        print("=" * 50)
        print("Gemini API Key:", self.api_key)
        print("=" * 50)

        self.client = genai.Client(
            api_key=self.api_key
        )

    def generate_answer(self, question, context):

        prompt = f"""
You are a helpful RAG assistant.

Use the provided document context to answer the user's question.

QUESTION:
{question}

DOCUMENT CONTEXT:
{context}

Give only the final answer in simple English.
"""

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt.strip()
        )

        return response.text