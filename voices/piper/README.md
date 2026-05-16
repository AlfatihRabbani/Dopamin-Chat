# Piper TTS voices

Drop `<voice>.onnx` + `<voice>.onnx.json` pairs here.

Get voices from [rhasspy/piper-voices on HuggingFace](https://huggingface.co/rhasspy/piper-voices). Pick a `*.onnx` file (e.g. `en_US-amy-medium.onnx`) and download both it and its sibling `.onnx.json`.

```bash
huggingface-cli download rhasspy/piper-voices \
    en/en_US/amy/medium/en_US-amy-medium.onnx \
    en/en_US/amy/medium/en_US-amy-medium.onnx.json \
    --local-dir voices/piper/
```

The launcher's **Text-to-speech** pane scans this folder. Voice is picked per-personality.
