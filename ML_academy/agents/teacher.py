import os

from memory.memory import save_topic
from dotenv import load_dotenv
from google import genai

from prompts.teacher_prompt import TEACHER_PROMPT

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def teacher_agent(question):

    # Temporary topic detection
    if "linear regression" in question.lower():
        save_topic("Linear Regression")

    prompt = f"""
    {TEACHER_PROMPT}

    Student Question:
    {question}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text

    except Exception as e:

        return f"""
Gemini API Error

{str(e)}

This is usually temporary server overload.
Try again in a few moments.
"""