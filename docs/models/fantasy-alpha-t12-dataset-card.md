# Fantasy Alpha T1.2 broad-plus-targeted dataset v2

This development-only mixture is designed to retain broad forecast behavior while
teaching short draft and tool actions. It contains 529 retained T1 traces and all
300 T1.1 traces. The 120 prompt identities corrected by T1.1 correspond to 156
old T1 sample rows; every old answer is superseded rather than duplicated. Another
126 stochastic teacher variants are deterministically deduplicated without outcomes.

- Rows: 829
- SHA-256: `39f4ccb5790263f2ae37151f31c6cda88a559742814da44ce5f5e0dc904c5d09`
- Supervision: last assistant response only
- Weighted learning signal: 70% broad retention / 30% targeted behavior
- Revision: tool-call share increased from 5% to 10% after the first canary
- Tool transitions: 30 complete call → result-conditioned pick groups
- Realized outcomes are not target inputs; 2025 remains sealed

## Schema allocation

| schema | rows | weighted assistant-token share |
|---|---:|---:|
| broad_bust | 59 | 17.5% |
| broad_full_slate | 155 | 17.5% |
| broad_season_threshold | 160 | 17.5% |
| broad_weekly_h2h | 155 | 17.5% |
| target_calibration | 120 | 5.0% |
| target_draft_pick | 96 | 7.0% |
| target_draft_recovery | 24 | 3.0% |
| target_post_tool_pick | 30 | 5.0% |
| target_tool_call | 30 | 10.0% |

## Important limitation

The draft labels imitate an outcome-blind rule policy. They can teach valid tool
use, legality, and roster discipline, but they do not prove superior player selection.
The frozen development evaluation must determine whether those behaviors transfer.
