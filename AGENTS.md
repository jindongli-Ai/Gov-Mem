# Workspace and storage requirements

User instruction (2026-09-19): ALL future code work, experiments, generated
outputs, caches and task temporary files must reside on `/mnt/data_disk_2`.
The old `/mnt/data_disk` filesystem is nearly out of inodes.

- Canonical project: `/mnt/data_disk_2/fuyali/codes/2027_TOIS_Gov-Mem`.
- Set this directory explicitly as the working directory for shell commands;
  a resumed Codex session may still inherit the old project working directory.
- Never write new files, run experiments, or edit the old project under
  `/mnt/data_disk/home/fuyali/codes/2027_TOIS_Gov-Mem`.
- Put task caches and temporary files under `/mnt/data_disk_2/fuyali/`.
  Do not follow old cache/output paths into `/mnt/data_disk` or use `/tmp`
  for experiment artifacts. Verify output/cache symlink targets before runs.
- Existing frozen reports/configs may contain historical old-disk paths.
  Preserve them for provenance; create new run configurations using new-disk
  paths rather than altering measured snapshots.
- Do not delete old project data or overwrite the migrated HOME implicitly.
- Paid module ablations are complete. Read `RESUME_GOVMEM.md` and
  `experiments/result/2026-09-19_Gov-Mem-v8_module_ablation_results.md`
  before continuing; do not rerun completed experiments unnecessarily.

## Experiment concurrency

User instruction (2026-09-20): Future experiments should start with 30 distinct
API keys in parallel by default, rather than 8 or 20. Explicitly configure the
launcher and verify the actual distinct-key/process count; do not assume a
worker argument bypasses older hard-coded concurrency caps. Use fewer workers
when fewer independent jobs remain, and preserve sequential history within
each episode. Do not log credentials. Preserve frozen run configurations and
snapshots. This instruction applies to future launches; the already running
Mem0 experiment remains at 20 unless separately changed.

## Local training and evaluation provenance

The user authorizes LoRA/MLP training when it addresses a demonstrated pipeline
need; A100 80GB GPUs are available. Training is optional, not a requirement to
add a neural component. Preserve symbolic/neuro-symbolic reasoning and evaluate
any trained component against the unchanged Gemini baseline with explicit cost
and error accounting.

Historical V7 runs already evaluated all 91 GateMem episodes. Do not describe
remaining dataset episodes as pristine holdout. The confirmation selection at
`experiments/manifests/v8_reserved_confirmation_episodes_20260919.json` must be
excluded from future training and teacher distillation; split by whole episode
or independent scenario family, never by checkpoints sharing history.

Read `docs/GOVMEM_V8_LOCAL_TRAINING_DECISION.md` before proposing a training run.
