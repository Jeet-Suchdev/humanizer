import httpx
import os
from dotenv import load_dotenv

load_dotenv()

async def chech_ai_score(text : str) -> dict:
    api_key = os.getenv("ZEROGPT_API_KEY")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.zerogpt.com/api/detect/detectText",
            headers = {
                "ApiKey" : api_key,
                "Content-Type" : "application/json"
            },
            json = {
                "input_text" : text
            },
            timeout = 30.0
        )

        data = response.json()

        ai_score = data.get("data", {}).get("aiScore", 0.0)
        fake_sentences = data.get("data", {}).get("fakeSentences", [])

        return {
            "ai_score" : float(ai_score),
            "fake_sentences" : fake_sentences,
            "is_ai" : ai_score > 5.0
        }
