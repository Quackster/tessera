# BUG.md: Flash-Next pipeline and long-context defects

This file tracks the open defects found while running Qwen3.8-Flash-Next
(IQ3_S) across two GPUs. Add one entry per defect. Keep the reproduction
command exact. Mark an entry fixed only after the full suite and an
end-to-end run pass.

## BUG-3: the pipeline diverges from the single-device engine

Status: open.

The two-stage pipeline reproduces the first few greedy tokens and then
diverges. The CLI loops on `!`. The opt-in test
`EngineTest.ShardedPipelineMatchesSingleDeviceWhenProvided` shows the
divergence directly.

Command:

```
tessera-cli run \
  --model ~/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --gpus 0,1 --pipeline --context 4096 --kv-q8 --max-completion-tokens 8 \
  --prompt-text "The capital of France is"
```

Symptom: the pipeline output is `!!!!!!!!` where the single-device engine
gives `The capital of France is **Paris**`. The gated test diverges at token
index 4 (pipeline `{760, 6511, 314, 9338, 369, 369, 369, 64}` versus engine
`{760, 6511, 314, 9338, 26458, 278, 26768, 284}`). The first four tokens
match.

Note: the gated test compares against a paged engine (`TESSERA_TEST_GPU_LAYERS`),
whose MoE experts partly run on the CPU. That engine is not a bit-exact
oracle for the full-GPU pipeline. The defect must be confirmed against a
same-path reference before it is fixed.

## BUG-1: Flash-Next MTP on the pipeline

Status: fixed for the load failure; the output defect is tracked as BUG-3.

The MTP head failed to load because `RunMtpDraft` needs the target
embedding, which a sharded load kept on the first stage only. Keeping the
embedding on the last stage when a drafter is attached fixed the load. The
route now runs but produces the BUG-3 output.

## BUG-2: DFlash2 on the pipeline fails device_error

Status: open.

Command:

```
tessera-cli run \
  --model ~/models/Qwen3.8-27B-MXFP4-MTPFP8 \
  --draft ~/models/Qwen3.8-27B-DFlash2-FP8 \
  --pipeline --gpus 0,1 --context 2048 --kv-q8 --max-completion-tokens 16 \
  --prompt-text "The capital of France is"
```

Symptom: Prepare now succeeds (the target embedding is on the last stage),
prefill runs, then the first speculative cycle fails device_error.

## 262144-token context on two R9700s (IQ3_S, kv-q8)

Status: fits, no OOM.

The QSA paged cache is fp16 and was fixed at 8192 cells. It now sizes to
the model context, so a 262144-token run allocates 262144 cells and a short
run keeps the old buffers.

Command:

```
tessera-cli run \
  --model ~/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --gpus 0,1 --pipeline --context 262144 --kv-q8 \
  --max-completion-tokens 8 --prompt-text "The capital of France is"
```

Measured, even 24/24 layer split, two R9700s (31.86 GiB usable each):

- stage 0 weights 24.16 GiB, stage 1 weights 26.88 GiB.
- QSA per stage about 3.6 GiB: the K and V pools are 512 MiB per full
  layer (6 full layers, 3.0 GiB), the indexer cos and sin tables 402 MiB,
  the pooled index 201 MiB.
- stage 0 about 27.8 GiB, stage 1 about 30.5 GiB. Both fit, so the full
  262144-token context loads without OOM.

The output correctness at this context is tracked as BUG-3.

## Largest even-split context with MTP (paged)

Status: fits, no OOM; MTP output correctness is BUG-1/BUG-3.

The even 24/24 split is memory-imbalanced: the last stage also carries the
output head, the token embedding, and the MTP head (about 3.35 GB). At
262144 tokens with MTP the last stage overflows, so offload the earliest
trunk layers with `--gpu-layers` (shard-local: each stage keeps its own
last N trunk layers).

Command:

```
tessera-cli run \
  --model ~/models/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
  --mtp ~/models/Qwen3.8-Flash-Next-MTP \
  --speculate --pipeline --pipeline-split 24,24 --gpus 0,1 \
  --context 262144 --kv-q8 --gpu-layers 18 --max-completion-tokens 12 \
  --prompt-text "The capital of France is"
```

`--gpu-layers 18` keeps each stage's last 17 trunk layers resident and
pages the rest. It loads and decodes without OOM (about 8 tok/s with MTP).
`--gpu-layers 20` (19 resident) still overflows the MTP head, so 18 or 19
is the working boundary.

Measured with `--gpu-layers 18`, even split, R9700 (31.86 GiB usable):

- stage 0 device weights 17.71 GiB (17 resident of 24 layers), stage 1
  device weights 19.74 GiB. The QSA cache adds about 3.59 GiB per stage.
- The paged layers (7 per stage, about 14 GiB total) stay in system RAM.
- The non-speculative paged run (`--gpu-layers 18`, no `--mtp`) at 262144
  prints `The capital of France is **Paris**.`, so paging preserves the
  result.

## Fixed

- DFlash2 Prepare returned malformed_file on the pipeline because the
  target embedding was on the first stage while the drafter runs on the
  last. Fixed by `ModelShard::last_stage_embedding` and
  `PipelineOptions::last_stage_embedding`.
- Flash-Next MTP on the pipeline ran out of memory on the last stage at an
  even split. Fixed by `PipelineOptions::stage_layers` and the CLI
  `--pipeline-split` flag.
- The QSA paged K/V pools were read before they were written after a
  layer split, so the sharded path was non-deterministic. Fixed by zeroing
  the pools at init.
