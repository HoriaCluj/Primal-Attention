---
name: vsc-account
description: User's VSC ID vsc39861, KU Leuven, credit account intro_vsc39861 (~2,000,000 intro credits as of 2026-11-16); login needs MFA SSH certificate
metadata:
  type: reference
---

- VSC ID: `vsc39861`, affiliated through KU Leuven (Horia, the thesis partner, is `vsc39859` / `intro_vsc39859`).
- Credit account: `intro_vsc39861`, confirmed by sam-balance on 2026-11-16: 2,000,000 credits. `SBATCH_ACCOUNT=intro_vsc39861` is set in ~/.bashrc on wICE.
- Paths: `$VSC_DATA=/data/leuven/398/vsc39861`, `$VSC_SCRATCH=/scratch/leuven/398/vsc39861`.
- Login: KU Leuven needs an MFA-issued SSH certificate (16 h), via `ssh -A vsc39861@login.hpc.kuleuven.be` and the browser link; non-managed laptops open https://firewall.vscentrum.be first. Plain keys (`~/.ssh/id_rsa_vsc`) are not enough.
- Docs: https://docs.vscentrum.be/ (MFA: accounts/mfa_login_firewall.html; credits: leuven/slurm_accounting.html).
- Scratch files not accessed for 30 days are purged.

Related: [[vsc-use-h100]].
- Since 2026-11-24 also in group/credit account `lp_robustness` (56.1M credits; shared lab project). Pass `sbatch --account=lp_robustness` on the command line: `SBATCH_ACCOUNT=intro_vsc39861` in ~/.bashrc overrides `#SBATCH --account`. gpu_h100: MaxTime 3 days, no per-user GPU cap (QOS normal).
