from openai import OpenAI


# Ollama exposes an OpenAI-compatible API on this local address by default.
client = OpenAI(
    base_url="http://127.0.0.1:11434/v1",
    api_key="ollama",
)

response = client.chat.completions.create(
    # Download this model first: `ollama pull qwen3:0.6b`.
    model="qwen3:0.6b",
    messages=[{"role": "user", "content": "Сколько будет 2 + 2. Ответь только 1 число"}],
    # Avoid spending tokens on a separate reasoning response for this tiny demo.
    reasoning_effort="none",
)

print(response.choices[0].message.content)
