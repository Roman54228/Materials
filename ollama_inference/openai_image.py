import base64
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# Read OPENAI_API_KEY from .env locally; .env must never be committed.
load_dotenv()

client = OpenAI()

# Resolve the image relative to this script so the command works from any directory.
IMAGE_PATH = Path(__file__).with_name("pasp.webp")
image = base64.b64encode(IMAGE_PATH.read_bytes()).decode()

response = client.responses.create(
    # Replace with a vision-capable model available to your OpenAI account if needed.
    model="gpt-5.6-terra",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Опиши, что изображено на картинке"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/webp;base64,{image}",
                },
            ],
        }
    ],
    # Image description does not require an extended reasoning response here.
    reasoning={"effort": "none"},
)

print(response.output_text)
