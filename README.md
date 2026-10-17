# Gov-Mem

This repository contains the code for **Gov-Mem**. The paper is currently **under review**.

Gov-Mem is a governance-aware **neuro-symbolic** framework for memory access and reasoning. It combines an incremental **Governed Slot Graph**, LLM-based semantic governance reasoning, event-state projection, a symbolic governance critic, and a safe-evidence boundary.

See [the framework-to-code correspondence](docs/FRAMEWORK.md) for the Fig. 2 execution path, [the method reference](docs/METHOD.md) for evaluation definitions, and [the file index](docs/CODE_INDEX.md) for the repository layout. The Governed Slot Graph is enabled in the default configuration and supplies advisory governance findings to the language reasoner; final disclosure is checked by the symbolic critic and safe-evidence boundary.

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

| Domain | Episodes | Queries |
|---|---:|---:|
| Medical | 21 | 579 |
| Office | 17 | 547 |
| Education | 30 | 540 |
| Household | 23 | 552 |
| **Total** | **91** | **2,218** |

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
