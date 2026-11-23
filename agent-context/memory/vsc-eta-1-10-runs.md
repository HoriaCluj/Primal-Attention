---
name: vsc-eta-1-10-runs
description: Pending wICE η=1/η=10 ImageNet-100 runs and spectra figure; full agent context lives in ~/thesis-workspace/agent-context/vsc-eta-1-10.md
metadata:
  node_type: memory
  type: project
  originSessionId: 77dae242-e88e-4767-8c1f-970d112df704
  modified: 2026-11-16T20:11:28.165Z
---

On 2026-11-16 the user set up wICE (vsc39861) and was about to queue the Primal.+DeiT-S ImageNet-100 runs at η = 1 and η = 10 (seed 0). The smoke test passed, the dry run looked correct, and actual submission is unconfirmed. Next steps: run the spectra job after the runs finish, get Horia's 4 `.npz` files, then replot `per_layer_c.png` with all six models.

**Why:** The user wants the η = 1 and η = 10 curves added to Horia's singular-value figure.
**How to apply:** When the user returns to the VSC experiments, read `~/thesis-workspace/agent-context/vsc-eta-1-10.md` first. It holds the paths, commands, status and gotchas. Start by asking for `sacct` / `log.txt` output to learn the real state. Related: [[vsc-account]], [[vsc-use-h100]].
