import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def coding_agent(query):

    prompt = f"""
    You are an expert Python, Machine Learning,
    and Deep Learning coding instructor.

    Explain code step by step.

    User Request:
    {query}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text