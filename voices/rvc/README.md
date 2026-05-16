# RVC fallback dir

The launcher scans this folder for legacy RVC layouts (a `<name>.pth` + same-stem `.index` directly under here, or one level deep). The preferred location is [`../../Applio/`](../../Applio).

`rmvpe.pt` (the pitch extractor) used to be downloaded here by an older code path. The current Applio integration places it inside `Applio_src/rvc/models/predictors/` — no manual download needed.
