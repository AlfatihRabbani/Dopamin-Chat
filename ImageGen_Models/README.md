# Image generation models

Drop diffusion checkpoints here. Supported families:

| Family | File type | Source |
|---|---|---|
| Stable Diffusion 1.5 / 2.x | `.safetensors` or HF folder | huggingface.co / civitai.com |
| SDXL | `.safetensors` or HF folder | huggingface.co / civitai.com |
| Flux.1 schnell / dev | HF folder | black-forest-labs |
| Qwen-Image-Edit | HF folder | Qwen org |
| LTX-Video | `.safetensors` | Lightricks |

The launcher's **Image generation** pane scans this directory at startup. Each subfolder or `.safetensors` file shows up as a selectable model.

For LoRAs, use the sibling [`ImageGen_LoRAs/`](../ImageGen_LoRAs) folder.

You can also point the UI at an external server (SD-WebUI / ComfyUI HTTP API) — no model files needed locally in that case.
