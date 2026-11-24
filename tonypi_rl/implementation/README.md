# Implementation specs

Hand these to a coding agent. Do not treat them as already applied until the file says so.

| Spec | Status | Train config |
|---|---|---|
| [NAV_V1_1.md](NAV_V1_1.md) | **implemented** (see [NAV_V1_1_IMPLEMENTATION_LOG.md](NAV_V1_1_IMPLEMENTATION_LOG.md)) | `tonypi_rl/configs/v1_1_nav_expert.yaml` |
| [NAV_V1_2.md](NAV_V1_2.md) | **implemented** (see [NAV_V1_2_IMPLEMENTATION_LOG.md](NAV_V1_2_IMPLEMENTATION_LOG.md)) | `tonypi_rl/configs/v1_2_nav_expert.yaml` |
| [NAV_V1_2_FIX.md](NAV_V1_2_FIX.md) | **implemented** (v1.2.1 patch, see [NAV_V1_2_IMPLEMENTATION_LOG.md](NAV_V1_2_IMPLEMENTATION_LOG.md) §6) | Resume clock `reset_num_timesteps=False` + remaining; hide ghost boxes; real u_trap; yaml `log_dir` |

Architecture background: [`../NAV.md`](../NAV.md).
