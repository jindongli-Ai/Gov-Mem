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

Edit `configs/govmem.yaml` to select the model, API endpoint, and retrieval `naive_top_k` (default: 20). Credentials are read from environment variables.

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
