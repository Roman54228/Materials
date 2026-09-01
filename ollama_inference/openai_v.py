from dotenv import load_dotenv
from openai import OpenAI


# Read OPENAI_API_KEY from .env locally; .env must never be committed.
load_dotenv()

client = OpenAI()

response = client.responses.create(
    # Replace with a model available to your OpenAI account if needed.
    model="gpt-5.6-luna",
    input="Сколько будет 2 + 2. Ответь только 1 число",
    # This deterministic arithmetic example does not need extra reasoning tokens.
    reasoning={"effort": "none"},
)

print(response.output_text)
