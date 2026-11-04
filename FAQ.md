# Frequenty Asked Questions

**Do I need to transfer my repository to Econ-ARK to get my REMARK listed?**

No. You keep your repository under your own GitHub account. You submit a pull request to the REMARK *catalog* repo that adds a single file (`REMARKs/your-project.yml`) pointing to your repo. On acceptance, Econ-ARK will create a *fork* of your repository to preserve the state at which it was tested and verified to work; the catalog and website will then point to that fork until you submit a new version. When you submit a new version, we will update the fork as long as `reproduce.sh` runs and the draft still meets REMARK requirements. You retain full ownership. See [STANDARD.md](STANDARD.md) § Submitting a REMARK.

Does inclusion of a REMARK in the catalog indicate that the Econ-ARK project endorses any concusions in it?

- No. This is a place for the material to be posted publicly for other people to see and form judgments on. Nevertheless, pull requests for attempted replications that are unsuccessful for unknown reasons will require a bit more attention from the Econ-ARK staff, which may include contacting the original author(s) to see if they can explain the discrepancies, or may include consulting with experts in the particular area in question.

