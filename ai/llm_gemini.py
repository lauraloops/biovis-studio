import os
from google import genai


class GeminiClient:
    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def generate(self, prompt: str, image_bytes: bytes) -> str:
        # Convert image bytes to base64 string for the API
        import base64
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        response = self.client.models.generate_content(
            model=self.model,
            contents={
                "role": "user",
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64
                        }
                    }
                ]
            }
        )
        return response.text or ""
