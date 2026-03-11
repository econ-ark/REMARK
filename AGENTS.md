# AGENTS.md -- Guidance for AI Coding Agents

This file is the entry point for AI coding agents (Cursor, Codex, Copilot, etc.)
working in the REMARK repository. For deeper context, see the `.agents/` directory.

## What This Repository Is

**REMARK** (R[eplications/eproductions] and Explorations Made using ARK) is a
**metadata-driven catalog** of reproducible computational economics research,
part of the [Econ-ARK](https://econ-ark.org/) project.

This repository does **not** contain the research code itself. It contains:

1. **Catalog entries** (`REMARKs/*.yml`) -- minimal YAML pointers to external
   GitHub repos that hold the actual research code.
2. **The REMARK Standard** (`STANDARD.md`) -- the canonical specification that
   research repos must satisfy for inclusion.
3. **Tooling** (`cli.py`) -- a CLI for cloning, linting, building, and executing
   the cataloged research projects.
4. **Documentation** -- guides for authors, editors, and maintainers.

## Econ-ARK Ecosystem

| Component | Role | Location |
|-----------|------|----------|
| **HARK** | Core Python toolkit for heterogeneous-agent models | [github.com/econ-ark/HARK](https://github.com/econ-ark/HARK) |
| **REMARK** (this repo) | Catalog + standard for reproducible research | [github.com/econ-ark/REMARK](https://github.com/econ-ark/REMARK) |
| **Individual REMARK repos** | Self-contained research projects | Linked from `REMARKs/*.yml` |
| **econ-ark.org** | Public website displaying the catalog | [econ-ark.org/materials](https://econ-ark.org/materials) |

The website is generated from this repo's catalog entries combined with
metadata (`CITATION.cff`, `REMARK.md`) from each individual research repo.
See `WORKFLOW.md` for the full pipeline.

## Repository Structure

```
REMARK/
├── AGENTS.md              # (this file) AI agent entry point
├── .agents/               # Detailed AI context, topics, schemas
│   ├── context.md         # Comprehensive repo context
│   ├── topics.md          # Research topic index and keywords
│   ├── api-guide.md       # Programmatic access patterns
│   └── schemas/           # Machine-readable metadata schemas
├── README.md              # Human-oriented overview
├── STANDARD.md            # Canonical REMARK requirements (3 tiers)
├── WORKFLOW.md            # Data flow: catalog -> website pipeline
├── cli.py                 # CLI tool for REMARK operations
├── REMARKs/               # YAML catalog entries (one per REMARK)
├── myst.yml               # MyST site configuration
├── templates/             # README templates for Tier 1/2/3
├── guides/                # Supplementary guides (e.g., journal-specific)
├── tools/                 # Developer utility scripts
└── assets/                # Images (logo, etc.)
```

## The Tier System

REMARKs are organized into three compliance tiers (defined in `STANDARD.md`):

| Tier | Name | Key Addition |
|------|------|-------------|
| **Tier 1** | Docker REMARK | Dockerfile + reproduce.sh + README (50+ lines) |
| **Tier 2** | Reproducible REMARK | + CITATION.cff + REMARK.md + README (100+ lines) |
| **Tier 3** | Published REMARK | + Zenodo DOI + specific git tag |

All tiers require: tagged release, `Dockerfile`, `reproduce.sh`, `LICENSE`,
`binder/environment.yml`.

## Common Pitfalls

- **This is a catalog, not a code repository.** The actual model code lives in
  external repos. Do not look for economic model implementations here.
- **`_REMARK/`** is a temporary workspace created by `cli.py`. Ignore it.
- **`REMARKs/` contains `.yml` files, not `.md` files.** The `.yml` files are
  the catalog entries; `.md` content lives in the individual research repos.
- **Two separate systems**: website generation (`populate_remarks.py` in the
  econ-ark.org repo) and REMARK validation (`cli.py` here) are independent.
  A repo can appear on the website without full REMARK compliance.

## Evaluating a Linked Draft REMARK

When the user has created a **symlink** in this repo to their draft REMARK
and asks you to evaluate it:

1. Read [STANDARD.md](STANDARD.md) and the contents of the linked directory.
2. For each tier (1, 2, 3), produce: (1) a checklist of requirements
   **already satisfied**, and (2) a checklist of **remaining items to do**.

See [guides/ai-compliance-check.md](guides/ai-compliance-check.md) for the
full prompt text and workflow so the user can reuse the same instructions
in another session if needed.

## Key Files for Understanding the System

| File | Purpose |
|------|---------|
| `STANDARD.md` | Canonical tier requirements -- start here for compliance rules |
| `WORKFLOW.md` | How catalog entries become website pages |
| `cli.py` | Tooling for pull/lint/build/execute |
| `REMARKs/*.yml` | The catalog itself |
| `How-To-Make-A-REMARK.md` | Author-facing guide |
| `EDITORIAL.md` | Editor workflow |

## Further AI Context

See `.agents/` for:
- `context.md` -- detailed repository context and integration points
- `topics.md` -- structured research topic index with keywords
- `api-guide.md` -- programmatic access patterns and code examples
- `schemas/` -- JSON schemas for REMARK metadata
