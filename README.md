# Gov-Mem

This repository contains the code for **Gov-Mem**. The paper is currently **under review**.

Gov-Mem governs memory-based language model agents through source-grounded state, claim review, and controlled evidence release.

See [the method reference](docs/METHOD.md) for the actual pipeline, four modules, evaluation definitions, and paper-writing guidance. See [the file index](docs/CODE_INDEX.md) for the repository layout.

## Setup

Use Python 3.10+ on Linux:

```bash
pip install -e ".[test]"
export OPENLUX_API_KEY="your-api-key"
```

The default model is `gpt-5.4-nano`. The experimental model set is GPT-5.4-nano, GPT-5-nano, GLM-4-AirX, and Kimi-K2 (`kimi-k2-0711-preview`).

Edit `configs/govmem.yaml` to select the model, API endpoint, and retrieval `naive_top_k` (default: 20). Credentials are read from environment variables.

## Dataset

We use the **GateMem dataset** to evaluate Gov-Mem across four domains: Medical, Office, Education, and Household (91 episodes and 2,218 queries). The dataset and official evaluation code are available in the [GateMem repository](https://github.com/rzhub/GateMem).

## Run

Obtain the [GateMem benchmark](https://github.com/rzhub/GateMem) separately. Point `--data_path` at a domain directory containing `episodes.jsonl` and `checkpoints.jsonl`.

```bash
python run_govmem.py \
  --dataset_name checkpoint_benchmark \
  --data_path /path/to/GateMem/gatemem/data/medical \
  --output_dir outputs/medical \
  --config configs/govmem.yaml \
  --skip_official_eval
```

For official scoring, set `GOVMEM_OFFICIAL_BENCHMARK_ROOT` to the GateMem checkout and omit `--skip_official_eval`. Install the benchmark's own dependencies as instructed upstream. Report U, A, F, and MGS per domain; Avg.MGS is the arithmetic mean of the four domain MGS values.

## Tests

```bash
pytest
```

## Acknowledgment

The code implementation for this study was assisted by Codex (GPT-6.1).
