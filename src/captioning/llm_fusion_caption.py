from dotenv import load_dotenv
import os
from google import genai
from google.genai import types


load_dotenv()  # loads .env
api_key = os.getenv("GEMINI_API_KEY")

class LLMSegmentCaptioner:
    def __init__(self, model: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def describe_segment(self, fusion_texts):
        joined = "\n\n".join(fusion_texts)
        prompt = (
            "You are analyzing a video segment.\n"
            "Using the following frame information, produce a single concise description "
            "that captures objects, actions, spatial relationships, and how things change over time.\n\n"
            f"FRAME INFORMATION:\n{joined}\n\n"
            "Segment description:"
        )

        # Create a chat (multi-turn) session
        chat = self.client.chats.create(model=self.model)

        resp = chat.send_message(prompt)
        return resp.text