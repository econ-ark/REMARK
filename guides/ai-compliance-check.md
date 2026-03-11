# Check Your Draft REMARK with an AI

Before submitting a pull request, you can have an AI evaluate your draft
REMARK and produce a tier-by-tier checklist of requirements already
satisfied and items still to do. The checklist aligns with the canonical
[STANDARD.md](../STANDARD.md) in this repository.

## Prerequisites

The AI must be able to read files in **both** repositories simultaneously:

- **This repository** (the REMARK catalog) -- so it can read `STANDARD.md`
  and the tier requirements.
- **Your draft REMARK repository** -- so it can inspect `Dockerfile`,
  `reproduce.sh`, `README.md`, `CITATION.cff`, `binder/environment.yml`,
  and other files the standard requires.

You therefore need an AI-assisted editor or coding environment that has
full file-system access to both repos at once -- for example Cursor,
Claude Code, GitHub Copilot, Windsurf, or similar.

The simplest way to give the AI access to both repos is to create a
**symbolic link** so that one repo appears as a subdirectory of the other.
Two strategies are described below; either works.

**Caveat:** In some AI-assisted editors, the background indexer does not
follow symbolic links. The symlinked directory may then be invisible to
search and to automatic context, even though the AI can still read
individual files if you point it at them explicitly (e.g. via `@` or
path references). If your tool behaves that way, the repo that is the
workspace root will be fully visible; the repo reached only through the
symlink may be only partially visible. That is why we offer two
strategies (draft-as-root vs. REMARK-as-root) so you can choose which
repo needs full visibility.

### Check that your setup gives the AI access to both repos

After you create the symlink but before you run the full compliance
prompt, you can ask your AI whether it can see both sides. Paste this
(adjust paths and names to match your setup):

```
I am preparing to run a REMARK compliance check. I have set up a
symbolic link so that one of two repositories appears inside the other.
Please confirm: (1) Can you list or read files from the REMARK catalog
repo (including STANDARD.md)? (2) Can you list or read files from my
draft REMARK repo (the one linked under [SYMLINK_NAME])? Tell me which
of these you can access and whether you need me to use a different
workspace layout or point you at specific paths.
```

Replace `[SYMLINK_NAME]` with the name of the symlink (e.g. `MyDraftRemark`
or `REMARK-catalog`). If the AI reports that it cannot see one of the
repos, try the other strategy or open both folders in a multi-root
workspace if your tool supports that.

## Strategy A: Symlink your draft INTO the REMARK clone (recommended)

This is the default recommendation because it also lets you run `cli.py
lint`, which expects to find the draft inside the REMARK working tree.

1. **Clone the REMARK repo** and `cd` into the clone:

   ```bash
   git clone https://github.com/econ-ark/REMARK.git
   cd REMARK
   ```

2. **Create a symbolic link** to your draft. Use as the link name the
   same `name` you will use in `REMARKs/{name}.yml`:

   ```bash
   ln -s /absolute/path/to/your-draft-remark MyDraftRemark
   ```

3. **Open the REMARK clone** as the workspace in your AI environment.
   The AI can now read `STANDARD.md` directly and inspect your draft
   under `MyDraftRemark/`.

4. **(Optional)** Add a temporary catalog entry so you can also run
   the CLI linter:

   ```bash
   cat > REMARKs/MyDraftRemark.yml <<EOF
   name: MyDraftRemark
   remote: https://github.com/your-org/your-repo
   title: Your Project Title
   EOF
   python cli.py lint REMARKs/MyDraftRemark.yml --tier N
   ```

## Strategy B: Symlink the REMARK clone INTO your draft repo

This alternative keeps your draft repo as the workspace root, which can
be useful if the AI needs deeper access to your code (some tools do not
fully index symlinked directories). The trade-off is that `cli.py lint`
will not work from this workspace because it expects the REMARK repo
as its working directory.

1. **Clone the REMARK repo** somewhere on your machine:

   ```bash
   git clone https://github.com/econ-ark/REMARK.git ~/REMARK
   ```

2. **In your draft repo**, create a symlink to the REMARK clone:

   ```bash
   cd /path/to/your-draft-remark
   ln -s ~/REMARK REMARK-catalog
   ```

3. **Open your draft repo** as the workspace. The AI can now read your
   files directly and access `REMARK-catalog/STANDARD.md` via the link.

## Which strategy to choose

| | Strategy A (REMARK is root) | Strategy B (draft is root) |
|-|---------------------------|--------------------------|
| **AI reads your draft** | Via symlink (may not be indexed by some tools) | Directly (full indexing) |
| **AI reads STANDARD.md** | Directly | Via symlink |
| **`cli.py lint` works** | Yes | No |
| **Best when** | You want both AI + CLI checks | Your tool does not index symlinks well |

If unsure, use **Strategy A**. If you find the AI cannot see your draft
files through the symlink, switch to Strategy B and point the AI at
`REMARK-catalog/STANDARD.md` explicitly.

## Prompt to Give the AI

After setting up either strategy, paste the following into your AI
assistant. Substitute **\[NAME]** with the symlink or directory name
for your draft (e.g. `MyDraftRemark` in Strategy A, or `.` in
Strategy B), and **\[TARGET_TIER]** with 1, 2, or 3.

```
I have created a symbolic link named [NAME] in this repository that
points to my draft REMARK. Please evaluate the repository at [NAME]
against the REMARK standard in STANDARD.md. For each of the three tiers
(1: Docker REMARK, 2: Reproducible REMARK, 3: Published REMARK),
produce: (1) a checklist of requirements already satisfied, and (2) a
checklist of remaining items to do. My target tier is [TARGET_TIER].
```

If using Strategy B, adjust the prompt to tell the AI where STANDARD.md
is (e.g. `REMARK-catalog/STANDARD.md`).

## What the Checklist Covers

The prompt is designed to produce, for each tier:

- **Satisfied** -- requirements from STANDARD.md that your draft already meets.
- **To do** -- requirements not yet met.

Summary of requirements (see [STANDARD.md](../STANDARD.md) for full text):

| Area | All tiers | Tier 1 | Tier 2 | Tier 3 |
|------|-----------|--------|--------|--------|
| **Base** | Tagged release; Dockerfile; reproduce.sh; README.md; LICENSE; binder/environment.yml | README >= 50 lines | README >= 100 lines; REMARK.md; CITATION.cff required | Tier 2 + REMARK.md with `tier: 3`; Zenodo DOI; git tag matching archive |
| **Optional** | reproduce_min.sh | CITATION.cff recommended | -- | -- |

The AI reads STANDARD.md and your repo to fill in the checklist; there
is no need to duplicate the full requirement text here.
