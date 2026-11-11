# Editorial Guidelines

The REMARK editor role is responsible for:

1. Guaranteeing that listed REMARKs are compliant with the
   [REMARK standard](STANDARD.md).

2. Adding the metadata needed for the REMARK to be published
   to the Econ-ARK website.

## Reviewing a Submission

When an author submits a pull request adding a new `REMARKs/{name}.yml`
catalog entry:

1. Verify the linked repository meets the requirements for the claimed
   tier (see [STANDARD.md](STANDARD.md)).
2. Run `python cli.py lint REMARKs/{name}.yml` to check compliance
   automatically.
3. Confirm `reproduce.sh` executes successfully in the provided
   environment.
4. Check that `CITATION.cff` and `REMARK.md` (if required by the
   target tier) are present and well-formed.

## Indexing a REMARK on the Website

The Econ-ARK website at [econ-ark.org/materials](https://econ-ark.org/materials)
is generated automatically from this repository's catalog entries combined
with metadata from each individual REMARK repository. See
[WORKFLOW.md](WORKFLOW.md) for the full pipeline.

To add a REMARK to the catalog, merge the PR that adds its
`REMARKs/{name}.yml` file. The PR will point to the author's repository.
On acceptance, the team will create a fork of the author's repo under the
econ-ark organization to preserve the state at which it was tested and
verified to work; the catalog (and website) will then point to that fork
until the author submits a new version. When the author submits a new
version, the team **will** update the fork as long as `reproduce.sh` runs
and the revised draft still meets REMARK requirements. The author keeps
ownership and can continue developing. The website will update within 24 hours via the automated
workflow.

Each catalog entry is a YAML file with at minimum:

```yaml
name: project-name
remote: https://github.com/org/repo
title: Human Readable Title
```

The website generation script (`populate_remarks.py` in the econ-ark.org
repo) reads `CITATION.cff` and `REMARK.md` from each linked repository
to build the material pages.

### Entries whose repository is not public yet

A REMARK can be listed before its repository is public, so that the catalog
stays the one list of every REMARK. Add `private: true` to its entry:

```yaml
name: project-name
remote: https://github.com/org/repo
title: Human Readable Title
private: true
```

- That the entry exists, with its name, title and remote, is public. Nothing
  else of the repository is, and the website shows nothing for the entry.
- `cli.py` leaves such entries out of `--all` unless `--private` is given,
  since most machines cannot reach them.
- A misspelled key would be read as absent, listing the entry as public. `lint`
  reports any key the catalog does not define, and the tests that run on a
  pull request to `REMARKs/` fail on one.
- When the repository becomes public, remove the line. Nothing else changes.
- The first version of a new REMARK may live under one account and later move
  to an author's. Update the entry's `remote` when it does; an entry is known
  by its `name`, which does not change.
