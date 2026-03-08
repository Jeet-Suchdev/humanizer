from langcgain_groq import ChatGroq
from langcgain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model = "llama-3,3-70b-versatile",
    api_key = os.getenv("GROQ_API_KEY"),
    temperature = 1.0
)

HUMANIZER_SYSTEM_PROMPT = """You are an expert academic writer who specializes in making 
AI-generated text sound completely natural and human-written.

Your job:
1. Rewrite the given text to sound like a real person wrote it
2. Keep the original meaning and all key facts EXACTLY the same
3. Use varied sentence lengths (mix short and long sentences)
4. Add natural transitions and connective phrases
5. Use contractions occasionally (it's, don't, we've)
6. Vary vocabulary — avoid repeating the same words
7. Add slight imperfections that humans naturally write
8. Maintain academic tone suitable for college assignments

CRITICAL: Do NOT change any facts, data, or core arguments. Only change the writing style.
Output ONLY the rewritten text, no explanations."""

humanizer_prompt = ChatPromptTemplate.from_messages(
    input_variables = ["text", "flagged_info"],
    validate_template = True,
    messages = [
        ("system", HUMANIZER_SYSTEM_PROMPT),
        ("human", "Rewrite this text to sound natural and human-written: {flagged_info}\n\n---\n{text}\n--")
    ]
)

async def humanize_text(text : str, flagged_sentences : list = None) -> str:
    max_flagged  = min(len(flagged_sentences), max(3, len(flagged_sentences) // 100))
    flagged_info = ""
    if flagged_sentences:
        flagged_info = f"\n\nPay special attention to rewriting these flagged sentences:\n"
        for i, sentence in enumerate(flagged_sentences[:max_flagged], 1):
            flagged_info += f"{i}. {sentence}\n"

    messages = humanizer_prompt.format_messages(
        text = text,
        flagged_info = flagged_info
    )

    response = await llm.ainvoke(messages)
    return response.content