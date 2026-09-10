# Workspace storage and synchronization

The currently used, validated payload is authoritative. A newer timestamp, a
branch name, or the presence of a local copy does not establish that authority.
Resolve conflicting physics files with the main analysis agent using the actual
production path and SHA-256 before synchronizing them.

| Location | Role and retention |
|---|---|
| `analysis/`, `autonomous_allhad/` source and configurations | Versioned implementation and dependencies. Match the adopted source and compiled payload hashes on EOS. |
| `analysis/data/{ids,corrections}.coffea` | Compiled physics inputs. Regenerate whenever their source changes; deploy source and payload together. |
| `analysis/hists/{btageff2024,topwtageff2024,topwtageff2025}.merged` | Protected required efficiency inputs; explicitly allowed in Git. |
| `autonomous_allhad/workflow/histograms/<campaign>/` | Campaign-local state, requests, validation and frozen code. Large cards, histograms, ROOT templates and fits live on EOS. |
| `autonomous_allhad/reports/` | Reviewed reports and compact exports. Do not remove evidence solely because it is old or untracked. |
| `docs/` | Public, reviewed website artifacts. Removed historical galleries link to their immutable Git snapshot and are labelled as archived. |
| `AN2019_016_v9.pdf` | Private reference, protected outside scratch and deliberately ignored by Git. `tmp/pdfs/AN2019_016_v9.pdf` is only a compatibility symlink. |
| `output/`, `tmp/` | Preview renders and working scratch. Builders may use these defaults; promote final handoffs explicitly to reports. The private-reference symlink must not be followed by cleanup. |
| `.workspace-local/` | Ignored local recovery records and conflicting-version snapshots. Never publish. |
| `.codex-worktrees/` | Ignored worktree area. Registered worktrees can contain independent work; never remove them by directory name alone. |
| `automation/`, `condor/`, `fast_analysis/`, `configs/stop_2024.yaml` | Retained legacy workflow and architecture evidence. CLI/config dependencies remain; absence of an import is not evidence that they are dead. |

The authoritative EOS repository location is documented in
[`workflow_inventory.md`](autonomous_allhad/reports/workflow_inventory.md).
Use matching repository-relative paths for source and cached results. This is a
source-and-manifest relationship, not a whole-directory mirror: ROOT data and
fit outputs stay on EOS, while private references and credentials stay out of Git.

Before evicting any local result, compare SHA-256 with the EOS copy and record
the relative restore path, size and hash inside its campaign. For the September
2026 campaign, see
[`local_storage_manifest_20260910.json`](autonomous_allhad/workflow/histograms/lepton_veto10_20260908/local_storage_manifest_20260910.json).
Its entries are local cache evictions, not failed jobs or lost production data.
Retain unmatched local files. Do not use recursive `rsync --delete`, `git clean`,
or directory-wide removal against an active analysis workspace.

Preserve campaign state and frozen code during source synchronization. Commit
only reviewed changes, push without force, and verify the deployed file hashes
against the recorded Git revision. A dirty EOS checkout must not be blindly
reset or switched to another branch.

The cleanup review and measured results are recorded in
[`workspace_cleanup_20260910/`](autonomous_allhad/reports/workspace_cleanup_20260910/).
