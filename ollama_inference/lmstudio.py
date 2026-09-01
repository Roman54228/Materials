from openai import OpenAI


# Start the Local Server in LM Studio's Developer tab before running this file.
client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
)

response = client.chat.completions.create(
    # Must match the model identifier exposed by the running LM Studio server.
    model="qwen/qwen3.5-9b",
    messages=[{"role": "user", "content": "Скажи привет"}],
    # Qwen's chat template supports turning off its thinking mode explicitly.
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)

print(response.choices[0].message.content)
