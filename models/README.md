# LLM models

Drop your chat model here.

## Supported formats

- **HuggingFace safetensors** — a folder containing `config.json`, `tokenizer*`, and `*.safetensors`.
- **GGUF** — a single `*.gguf` file (sharded GGUFs also work).

## Examples

```bash
# Safetensors (full precision, will be 4-bit quantized at load)
huggingface-cli download google/gemma-3-12b-it --local-dir models/gemma-3-12b-it

# GGUF (already quantized — best for low-VRAM)
huggingface-cli download unsloth/gemma-3-12b-it-GGUF \
    "gemma-3-12b-it.Q4_K_M.gguf" --local-dir models/
```

Multimodal variants (Gemma-3-VL, Qwen-VL, Llama-3.2-Vision) are detected automatically — image attachments will be passed to the LLM when `vision_enabled: true` in the personality.

The launcher's chat UI scans this folder and shows a picker if multiple models exist.
