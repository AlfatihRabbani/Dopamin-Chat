# RVC voice models

Drop Applio-compatible RVC v2 models here. Layout:

```
Applio/
└── <character_name>/
    ├── <character_name>.pth
    └── <character_name>.index
```

Sources for `.pth` + `.index` pairs:

- [weights.gg](https://www.weights.gg/) — large public catalogue
- Applio's own community model index ([Applio docs](https://docs.applio.org/))
- Train your own with Applio's training pipeline

The launcher's **Voice conversion (RVC)** pane scans this folder and pairs each `.pth` with its same-stem `.index`.

The first inference will lazily download Applio's required prerequisites (rmvpe pitch extractor, contentvec embedder, fcpe) into `Applio_src/rvc/models/` — about 600 MB. Subsequent runs use the cached copies.
