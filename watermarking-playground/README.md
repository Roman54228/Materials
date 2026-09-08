---
title: Watermarking Playground
emoji: 🐨
colorFrom: yellow
colorTo: purple
sdk: static
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## Checking pasted text

The **Check a text** panel runs the same detector used for generated output:

- **Hard** and **Soft** use the keyed green-list z-test from Kirchenbauer et al. (2023).
- **Tournament** uses the keyed g-value test used by the playground's SynthID-Text-style sampler.

Paste text, select the same model and watermarking settings used by its source, and enter the corresponding detector key. The first `h` tokens are used only as context and are excluded from scoring, since a standalone pasted string has no preceding prompt. The result is meaningful only when the tokenizer, scheme, key, `gamma`/`h` (or `m`/`h`), and optional forced-red words match the source watermark configuration.
