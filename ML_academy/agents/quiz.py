import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


def quiz_agent(topic):

    prompt = f"""
    Create a Machine Learning quiz on:

    {topic}

    Include:

    - 5 MCQs
    - Answers
    - Difficulty: Beginner
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text