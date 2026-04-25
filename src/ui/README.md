---
title: LEO Translation Hub
emoji: 🦁
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.16.0
app_file: hf_spaces_app.py
pinned: false
license: mit
---

# 🦁 L.E.O. Translation Hub

This Space hosts the **Linguistic Engineering Optimization** (LEO) translation model, based on **Seamless-M4T v2 Large** with LoRA adapters specialized for **Roverplastik** technical terminology (Italian -> English, French, Spanish).

## 🚀 Deployment Config

To run this Space, ensure you have set the `ADAPTER_PATH` environment variable to point to the hosted LoRA adapters on the Hugging Face Hub (e.g., `maxbsdv/LEO-SeamlessM4T-v2-Large-Roverplastik`), or upload the adapter files directly to the root of this Space.

**Recommended Setup:**
1. Host the model in a separate Hugging Face Model repository.
2. Set `ADAPTER_PATH` in the Space's "Settings" -> "Variables and secrets" to your model ID.
