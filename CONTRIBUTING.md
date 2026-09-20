# Contributing to REMARK

Thank you for your interest in contributing to the REMARK catalog!

## Submitting a New REMARK

If you have a reproducible computational economics project you would like
to add to the catalog:

**Check your draft before submitting.** You can have an AI generate a
tier-by-tier compliance checklist before you submit. The AI needs access
to both your draft repo and this REMARK repo simultaneously; the guide
explains two symlink strategies and includes a copy-paste prompt. See
**[Check your draft with AI](guides/ai-compliance-check.md)**.

1. Ensure your project meets the requirements in [STANDARD.md](STANDARD.md)
   for your target tier.
2. Follow the step-by-step instructions in
   [QUICK-START.md](QUICK-START.md) or [How-To-Make-A-REMARK.md](How-To-Make-A-REMARK.md).
3. Submit a pull request adding a `REMARKs/{your-project}.yml` catalog
   entry to this repository. The entry points to **your** repository (your
   GitHub identity); you keep ownership. On acceptance, Econ-ARK will fork
   your repo to preserve the state at which it was tested and verified to
   work; the catalog and website will then point to that fork until you
   submit a new version. When you submit a new version, Econ-ARK will
   update the fork as long as `reproduce.sh` runs and the draft still
   meets REMARK requirements. See [STANDARD.md](STANDARD.md) § Submitting a REMARK.

## Improving This Repository

Contributions to the REMARK standard, documentation, and tooling are
welcome. Please open an issue first to discuss significant changes.

## Questions

- Check [FAQ.md](FAQ.md) for common questions.
- Open a [GitHub issue](https://github.com/econ-ark/REMARK/issues) for
  anything else.
