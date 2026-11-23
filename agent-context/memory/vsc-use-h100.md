---
name: vsc-use-h100
description: "On VSC wICE, run all jobs (training and spectra) on the gpu_h100 partition, not gpu_v100"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 77dae242-e88e-4767-8c1f-970d112df704
  modified: 2026-11-16T19:13:32.292Z
---

Run every VSC wICE job on H100 (`gpu_h100`), including the spectra job that the handoff suggests sending to `gpu_v100`.

**Why:** User decided this on 2026-11-16 after seeing the H100 credit costs (~150k credits/h for 4 H100 + 64 cores, ~37.5k/h for 1 H100 + 16 cores).
**How to apply:** Give `sbatch` commands without the `--partition=gpu_v100` override; the `jobs/*.slurm` scripts already default to `gpu_h100`. Related: [[primal-attention-eta-ablation]].
