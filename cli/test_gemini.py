#
import os

from dotenv import load_dotenv
from google import genai

# load api key
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
# load model
client = genai.Client(api_key=api_key)
content_string = (
    "Why is Boot.dev such a great place to learn about RAG? Use one paragraph maximum."
)
mresp = client.models.generate_content(
    model="gemini-2.5-flash", contents=content_string
)
print(f"\nmodel response:\n{mresp.text}")
print(f"Prompt Tokens: {mresp.usage_metadata.prompt_token_count}")
print(f"Response Tokens: {mresp.usage_metadata.candidates_token_count}\n")
