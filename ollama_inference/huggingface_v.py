"""Локальный запуск Qwen3-0.6B из Hugging Face.

Установка:
    pip install "transformers>=4.51.0" torch

При первом запуске модель автоматически скачается в кэш Hugging Face.
Следующие запуски будут использовать уже скачанные файлы.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen3-0.6B"

# Prefer a GPU when one is available; MPS is PyTorch's Apple Silicon backend.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# The first call downloads model files into Hugging Face's local cache.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
).to(device)

messages = [
    {"role": "user", "content": "Сколько будет 2 + 2? Ответь только одним числом."}
]

# Format messages using the model's native chat template before tokenization.
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True,
).to(model.device)

with torch.inference_mode():
    # Inference mode avoids storing gradients and reduces memory use at generation.
    output = model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
    )

# `generate` returns the prompt plus completion; print only newly generated tokens.
new_tokens = output[0, inputs["input_ids"].shape[1] :]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
