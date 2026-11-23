---
name: in1k-full-runs-etas
description: "The 3 full 300-epoch ImageNet-1K runs are DeiT, Primal η=0.05 and Primal η=0.5 (η=0.05 replaces η=0)"
metadata:
  node_type: memory
  type: project
  originSessionId: 77dae242-e88e-4767-8c1f-970d112df704
  modified: 2026-11-23T10:20:15.416Z
---

The full ImageNet-1K runs are DeiT-S, Primal η = 0.05 and Primal η = 0.5. Use η = 0.05, not η = 0, even though the 2026-11-23 smoke test used η = 0.

**Why:** the user's decision on 2026-11-23 after the smoke tests; no reason was given.

**How to apply:** when submitting the full runs, pass `ETA=0.05` (run name `primal_eta0.05_seed0`). Speed and cost match the η = 0 smoke test: about 13.2 h and 2.0M credits per run. See [[vsc-eta-1-10-runs]] and [[vsc-use-h100]].
