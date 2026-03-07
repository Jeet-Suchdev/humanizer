from langchain_groq import ChatGroq
from lkangchain_core.messages import HumanMessage, SystemMessage
import json
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
  model = "llama-3.1-70b-versatile",
  api_key = os.getenv("GROQ_API_KEY"),
  temperature = 0.3
)

CRITIC_SYSTEM_PROMPT = """You are a strict academic writing evaluator at a top university.
Evaluate the given text for academic suitability for college assignments or research papers.

Check for:
1. Proper academic tone (formal, not casual)
2. Clear argument structure
3. Logical flow and coherence
4. Appropriate vocabulary level
5. No grammatical errors
6. Professional presentation

Respond ONLY with a JSON object in this exact format:
{
  "is_academically_suitable": true or false,
  "score": 0-100,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"],
  "overall_feedback": "brief overall comment"
}"""

async def check_academic_quality(text : str) -> dict:
  messages = [
    SystemMessage(content = CRITIC_SYSTEM_PROMPT),
    HumanMessage(content = f"Evaluate this text  fro academic suitability:\n\n{text}")
  ]

  response = await llm.ainvoke(messages)
  try:
    content = response.content.strip()
    if content.startswith("'''"):
      content = content.split("'''")[1]
      if content.startswith("json"):
        content = content[4:]
    result = json.loads(content)
  except json.JSONDecodeError:
    result = {
      "is_academically_suitable": True,
            "score": 75,
            "issues": [],
            "suggestions": [],
            "overall_feedback": response.content
    }

  return result
    