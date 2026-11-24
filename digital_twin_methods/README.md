# Selectable digital twin methods

This directory contains adapters to the research implementations. The existing four benchmark runners and active baseline experiment are unchanged. All method calls use separate per-user/per-trial state; the downstream model, official question format and evaluator remain the responsibility of the benchmark runner.

## Source versions

| Method | Source and fixed checkout |
|---|---|
| GPG | [Paper, §3 and §5.2](https://aclanthology.org/2024.findings-emnlp.231/). Paper-derived workflow; task-specific questions are adapted to the four benchmarks. |
| AlignXplore+ | [AlignX-Family](https://github.com/AntResearchNLP/AlignX-Family), `b9b8a9ebaaa2b83bd55d40a5d25e7587e8c6944d`; official streaming prompt function and [trained checkpoint](https://huggingface.co/VanillaH1/AlignXplore-Plus). |
| PGMem | [pgmem](https://github.com/wonjunchoi23/pgmem), `879a7a964f898c2f654977f871041bf20d09c7ca`; official PersonaMem graph/updater/retriever. |
| O-Mem | [O-Mem](https://github.com/OPPO-PersonalAI/O-Mem), `46e131ac39af55d456304c61dfb881717044528e`; official working, episodic and persona memory, with local Windows/import and bounded-retry edits. |
| Mem0 | [mem0 v0.1.94](https://github.com/mem0ai/mem0/tree/07ddd7cb4bd67962cf9a988d7b5c3f3920fad2d4); official ADD/UPDATE/DELETE/NONE vector memory algorithm. |

## Environment and entry points

Run from the project root. The independently installed `.deps` directory is used for the additional packages; the existing FingerTip environment provides Torch and sentence-transformers. No dependencies were installed into the active Vita environment. PGMem and O-Mem use the downloaded official `all-MiniLM-L6-v2` encoder. Mem0 uses local Qdrant and SQLite with telemetry disabled and the existing loopback embedding service (a marked replacement for the original OpenAI embedder). AlignXplore+ requires its dedicated checkpoint and a compatible inference environment; the trained model is not bundled.

```powershell
$env:PYTHONPATH=(Join-Path (Get-Location) 'digital_twin_methods/.deps')
.fingertip-venv/Scripts/python.exe -m digital_twin_methods.test_methods
```

The test uses synthetic histories and mocked LLM outputs. It exercises real source code for graph and vector storage, not live Nex or AlignXplore+ inference.

Use Python for reusable memory:

```python
from digital_twin_methods import create, History

memory = create('gpg', task='lamp7')
memory.update([History('I LOVE walking!', timestamp='2025-01-01')])
context = memory.retrieve('Rewrite this tweet')
# Insert context into the benchmark's existing history input slot.
```

Select `gpg`, `alignxplore_plus`, `pgmem`, `omem`, or `mem0`. O-Mem and Mem0 require unique `storage` per user/trial; AlignXplore+ requires `generator=AlignGenerator(checkpoint_path)` (optional `quantize=True`, requiring a CUDA/quantization-compatible environment). For stateless evaluations the CLI also accepts `--history`, `--query`, `--method`, and `--output`; use `--help` to list other settings.

The model transport reuses the fixed Nex medium (60 s) → none (30 s) policy and DNS backoff for construction calls. PGMem JSON is parsed and validated against the upstream schema. The API generation cap is separate from the desired summary size; benchmark context limits must still be applied at the final prompt. The original Mem0 package's default nearest-memory candidate count is five; the paper's ten-candidate/window/global-summary variant is not represented as a completed reproduction.

## Prepared benchmark inputs

`bench_inputs.py` exposes `lamp_history`, `mega_history`, `fingertip_history`, and `vita_history`. Feed only the original visible history, not the current target answer. FingerTip selects only prior timestamps and defaults to the original recent-20 setting. Mega uses Twin's historical `persona_json`; it does not mark unselected answers as rejected choices. AlignXplore+ consumes explicit query/chosen/rejected triples and leaves missing negative evidence unspecified.

Vita supports direct custom class selection via `--memory-class digital_twin_methods.vita_adapter.GPGMemory` (and `AlignXploreMemory`, `PGMemMemory`, `OMemMemory`, `Mem0Memory`) once the chosen method's dependencies are available in that Python environment. The adapter has no effect until selected. The AlignXplore+ adapter uses `ALIGNXPLORE_CHECKPOINT` and optional `ALIGNXPLORE_4BIT=1`. Validate each new backend with a small full-user sequence before official runs. LaMP, Mega and FingerTip can call the same factory at their current history construction stage; the existing formal runners have not yet been switched or tested end to end with these methods.
