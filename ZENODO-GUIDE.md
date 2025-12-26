# Zenodo DOI Guide for REMARKs

**Complete step-by-step guide for obtaining a permanent Zenodo DOI for your REMARK**

For Published REMARKs, a permanent Zenodo DOI is required. This guide walks you through the entire process.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Checklist](#quick-checklist)
3. [Detailed Step-by-Step Instructions](#detailed-step-by-step-instructions)
4. [Troubleshooting](#troubleshooting)
5. [Verification](#verification)
6. [FAQ](#faq)

---

## Prerequisites

Before starting, ensure you have:

- ✅ A REMARK repository that meets Standard REMARK requirements
- ✅ GitHub repository is public (Zenodo cannot archive private repos)
- ✅ No uncommitted changes in your repository
- ✅ `CITATION.cff` file with complete metadata
- ✅ GitHub account with admin access to the repository
- ✅ Zenodo account (free - create at https://zenodo.org using GitHub login)

---

## Quick Checklist

Use this checklist to track your progress:

### Phase 1: Pre-Flight (5 minutes)
- [ ] 1.1 Verify repository is clean (no uncommitted changes)
- [ ] 1.2 Verify CITATION.cff exists and is complete
- [ ] 1.3 Test `reproduce.sh` runs successfully
- [ ] 1.4 Ensure all required files are present

### Phase 2: Git Tagging (2 minutes)
- [ ] 2.1 Create annotated git tag (e.g., `v1.0.0`)
- [ ] 2.2 Push tag to GitHub

### Phase 3: Zenodo Setup (5-10 minutes)
- [ ] 3.1 Log in to Zenodo with GitHub account
- [ ] 3.2 Navigate to GitHub settings on Zenodo
- [ ] 3.3 Enable Zenodo-GitHub integration
- [ ] 3.4 Find and toggle your repository on Zenodo
- [ ] 3.5 Sync repositories if needed

### Phase 4: GitHub Release (2 minutes)
- [ ] 4.1 Go to GitHub repository releases page
- [ ] 4.2 Click "Draft a new release"
- [ ] 4.3 Select your tag from dropdown
- [ ] 4.4 Add release title and description
- [ ] 4.5 Publish release

### Phase 5: Zenodo Archival (5-10 minutes)
- [ ] 5.1 Wait for Zenodo to archive (automatic)
- [ ] 5.2 Find your upload on Zenodo
- [ ] 5.3 Obtain DOI from Zenodo page
- [ ] 5.4 Copy DOI badge markdown

### Phase 6: Update CITATION.cff (2 minutes)
- [ ] 6.1 Add DOI to CITATION.cff
- [ ] 6.2 Commit and push update
- [ ] 6.3 Verify DOI badge appears on GitHub

### Phase 7: Verification (5 minutes)
- [ ] 7.1 Download archive from Zenodo
- [ ] 7.2 Verify files are complete
- [ ] 7.3 Test basic functionality
- [ ] 7.4 Verify commit hash matches

**Total Time**: 30-45 minutes

---

## Detailed Step-by-Step Instructions

### Step 1: Pre-Flight Checks (5 minutes)

#### 1.1 Verify Repository is Clean

```bash
cd /path/to/your-remark
git status
```

**Expected output**: `nothing to commit, working tree clean`

If you see uncommitted changes, commit them first:

```bash
git add .
git commit -m "Prepare for v1.0.0 release"
git push origin main  # or master
```

#### 1.2 Verify CITATION.cff

Check that your CITATION.cff has all required fields:

```bash
cat CITATION.cff
```

**Required fields**:
- `cff-version` (e.g., "1.2.0")
- `title`
- `authors` (with names)
- `repository-code` (GitHub URL)
- `license` (e.g., "Apache-2.0")

**Helpful but not required**:
- `keywords`
- `abstract`
- ORCIDs for authors

#### 1.3 Test Reproduction Script

```bash
./reproduce.sh --help
```

Should display help information without errors.

#### 1.4 Verify Required Files

```bash
ls -la reproduce.sh CITATION.cff binder/environment.yml LICENSE
```

All should exist and be readable.

---

### Step 2: Create Git Tag (2 minutes)

#### 2.1 Create Annotated Tag

Choose a version number following [semantic versioning](https://semver.org/):
- **v1.0.0** for first release
- **v1.1.0** for minor updates
- **v1.0.1** for bug fixes

```bash
git tag -a v1.0.0 -m "First published release"
```

**Verify tag was created**:

```bash
git tag -l
git show v1.0.0
```

#### 2.2 Push Tag to GitHub

```bash
git push origin v1.0.0
```

**Verify on GitHub**:
- Go to: `https://github.com/YOUR_USERNAME/YOUR_REMARK/tags`
- Your tag should appear in the list

---

### Step 3: Enable Zenodo-GitHub Integration (5-10 minutes)

#### 3.1 Log in to Zenodo

1. Go to https://zenodo.org
2. Click "Log in" (top right)
3. Select "Log in with GitHub"
4. Authorize Zenodo if prompted

#### 3.2 Navigate to GitHub Settings

1. Click your username (top right)
2. Select "GitHub" from dropdown menu
3. This opens: https://zenodo.org/account/settings/github/

#### 3.3 Grant Repository Access

**If this is your first time using Zenodo**:

1. You'll see a message: "Authorize application access on GitHub"
2. Click the link to GitHub settings
3. On GitHub, find "Zenodo" in Applications
4. Click "Configure"
5. Under "Repository access":
   - Select "Only select repositories"
   - Choose your REMARK repository
6. Click "Save"

**Return to Zenodo** and refresh the GitHub settings page.

#### 3.4 Enable Your Repository

On the Zenodo GitHub settings page:

1. Find your repository in the list
2. Toggle the switch to **ON** (green)
3. If you don't see your repository, click "Sync now"

**Common issues**:
- Repository not appearing? Click "Sync now" and wait 30 seconds
- Toggle doesn't work? Check GitHub authorization settings
- Still not working? See [Troubleshooting](#troubleshooting)

---

### Step 4: Create GitHub Release (2 minutes)

#### 4.1 Navigate to Releases

Go to: `https://github.com/YOUR_USERNAME/YOUR_REMARK/releases`

#### 4.2 Draft New Release

1. Click "Draft a new release" button

#### 4.3 Fill in Release Details

**Choose a tag**: Select `v1.0.0` from dropdown

**Release title**: Use a clear, descriptive title
- Good: "Initial Published Release"
- Good: "Version 1.0.0 - QE Submission"
- Avoid: Just "v1.0.0"

**Description**: Include key information (example):

```markdown
## Version 1.0.0 - Initial Published Release

This release accompanies the submission to [Journal Name].

### What's Included
- Complete replication package
- All data and code
- Comprehensive documentation
- Automated reproduction workflow

### System Requirements
- See `binder/environment.yml` for dependencies
- Reproduction time: ~X minutes on standard laptop

### Citation
See CITATION.cff for bibliographic information.

### License
Apache License 2.0
```

#### 4.4 Publish Release

1. Leave "This is a pre-release" **unchecked**
2. Click "Publish release"

**Verification**:
- Release appears at: `https://github.com/YOUR_USERNAME/YOUR_REMARK/releases/tag/v1.0.0`
- You'll receive a GitHub notification

---

### Step 5: Obtain Zenodo DOI (5-10 minutes)

#### 5.1 Wait for Zenodo Archival

Zenodo automatically archives your release when you publish it.

**Timeline**:
- Archival starts: Immediately after release
- Processing time: 2-10 minutes
- Factors: Repository size, Zenodo server load

**Monitoring**:

1. Go to Zenodo uploads: https://zenodo.org/deposit
2. Look for your repository name
3. Status will show "Processing..." then "Published"

**Alternative**: Check your email for Zenodo notification

#### 5.2 Find Your Upload

1. Go to: https://zenodo.org/deposit
2. Click on your REMARK name
3. This opens the Zenodo landing page

**Or navigate directly** (after it's published):
- Search for your repository name on https://zenodo.org

#### 5.3 Obtain the DOI

On the Zenodo landing page:

1. **DOI badge**: At top right of page
2. **DOI format**: `10.5281/zenodo.XXXXXXX` (7-digit number)

**Copy the DOI number** (just the number, e.g., `10.5281/zenodo.1234567`)

#### 5.4 Get DOI Badge Markdown

On the same Zenodo page:

1. Look for "DOI" badge (top right)
2. Click the badge
3. Select "Markdown" format
4. Copy the code, which looks like:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)
```

---

### Step 6: Update CITATION.cff (2 minutes)

#### 6.1 Add DOI to CITATION.cff

Edit your `CITATION.cff` file and add the DOI:

```yaml
cff-version: 1.2.0
title: "Your REMARK Title"
authors:
  - family-names: "Smith"
    given-names: "John"
repository-code: 'https://github.com/YOUR_USERNAME/YOUR_REMARK'
doi: 10.5281/zenodo.1234567  # ADD THIS LINE
license: Apache-2.0
keywords:
  - REMARK
  - Reproducibility
```

**Important**: 
- Place DOI after `repository-code`
- Use the **plain DOI**, not the full URL
- Format: `doi: 10.5281/zenodo.XXXXXXX`

#### 6.2 Commit and Push

```bash
git add CITATION.cff
git commit -m "Add Zenodo DOI to CITATION.cff

DOI: 10.5281/zenodo.XXXXXXX
Archive: https://doi.org/10.5281/zenodo.XXXXXXX

Published REMARK complete."

git push origin main  # or master
```

#### 6.3 Optional: Add DOI Badge to README

Edit your `README.md` and add the badge at the top:

```markdown
# Your REMARK Title

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1234567.svg)](https://doi.org/10.5281/zenodo.1234567)

[Rest of README...]
```

---

### Step 7: Verification (5 minutes)

#### 7.1 Download Archive from Zenodo

Test that anyone can download your archive:

```bash
# Get download URL from Zenodo page, then:
cd /tmp
curl -L -O "https://zenodo.org/records/XXXXXXX/files/YOUR_USERNAME-YOUR_REMARK-YYYYYYY.zip"
```

#### 7.2 Verify Archive Contents

```bash
unzip YOUR_USERNAME-YOUR_REMARK-YYYYYYY.zip
cd YOUR_USERNAME-YOUR_REMARK-YYYYYYY
ls -la
```

**Check for**:
- `reproduce.sh` exists and is executable
- `binder/environment.yml` exists
- `CITATION.cff` exists
- All code and data files present

#### 7.3 Test Basic Functionality

```bash
./reproduce.sh --help
```

Should display help without errors.

#### 7.4 Verify Commit Hash Matches

```bash
# In the Zenodo download directory:
git rev-parse HEAD

# In your original repository:
cd /path/to/your-remark
git rev-parse v1.0.0
```

**Both should show the same commit hash** - this is cryptographic proof they're identical.

---

## Troubleshooting

### Problem: Repository Not Appearing on Zenodo

**Symptoms**: Your repository doesn't appear in Zenodo's GitHub settings page

**Solutions**:

1. **Click "Sync now"** on Zenodo GitHub settings page
   - Wait 30-60 seconds
   - Refresh page

2. **Check GitHub authorization**:
   - Go to https://github.com/settings/installations
   - Find "Zenodo"
   - Click "Configure"
   - Verify your repository is selected under "Repository access"
   - Save changes

3. **Check repository is public**:
   - Zenodo cannot access private repositories
   - Go to repository settings on GitHub
   - Under "Danger Zone", ensure it's public

4. **Check organization permissions** (if applicable):
   - Organization settings → Third-party access
   - Ensure Zenodo is authorized
   - Grant access to repositories

### Problem: Release Created but No Zenodo Archive

**Symptoms**: GitHub release exists, but no Zenodo archive created

**Solutions**:

1. **Verify Zenodo toggle is ON**:
   - Go to https://zenodo.org/account/settings/github/
   - Find your repository
   - Ensure toggle is green (ON)

2. **Check release is not a "pre-release"**:
   - Pre-releases don't trigger Zenodo archival
   - Edit release on GitHub
   - Uncheck "This is a pre-release"

3. **Wait longer**:
   - Large repositories take longer to archive
   - Check back in 15-30 minutes

4. **Create a new release**:
   - Sometimes the webhook fails
   - Create a new tag: `v1.0.1`
   - Create a new release
   - Should trigger archival

### Problem: DOI Badge Not Showing on GitHub

**Symptoms**: Added DOI to CITATION.cff but badge doesn't appear

**Solutions**:

1. **Verify CITATION.cff syntax**:
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('CITATION.cff'))"
   ```

2. **Check DOI format**:
   - Correct: `doi: 10.5281/zenodo.1234567`
   - Incorrect: `doi: https://doi.org/10.5281/zenodo.1234567`

3. **Wait for GitHub cache**:
   - GitHub caches CITATION.cff
   - Wait 5-10 minutes
   - Hard refresh browser (Ctrl+F5 or Cmd+Shift+R)

### Problem: Archive Missing Files

**Symptoms**: Downloaded Zenodo archive is incomplete

**Solutions**:

1. **Check .gitignore**:
   - Files in `.gitignore` won't be archived
   - Review and update `.gitignore`
   - Create new tag and release

2. **Verify files were committed**:
   ```bash
   git ls-tree -r v1.0.0 --name-only
   ```
   - This shows all files in the tag

3. **Large files**:
   - GitHub releases have file size limits
   - Consider data download instructions instead
   - Use Git LFS for large files

---

## Verification Checklist

After completing all steps, verify:

### On GitHub
- [ ] Tag `v1.0.0` exists in tags list
- [ ] Release published with tag `v1.0.0`
- [ ] CITATION.cff contains DOI
- [ ] DOI badge appears in repository sidebar

### On Zenodo
- [ ] Repository archived and published
- [ ] DOI assigned and visible
- [ ] Landing page shows correct information
- [ ] Archive can be downloaded

### Cryptographic Verification
- [ ] Commit hash from Zenodo matches tag
- [ ] Downloaded archive extracts successfully
- [ ] `reproduce.sh` exists and is executable

### Documentation
- [ ] README mentions Zenodo archive
- [ ] CITATION.cff has DOI
- [ ] Release notes are clear and complete

---

## FAQ

### Q: Do I need to create a new DOI for every version?

**A**: No. Zenodo automatically creates version-specific DOIs, but all versions share a "concept DOI" that always points to the latest version. You can use either:
- **Version-specific DOI**: `10.5281/zenodo.1234567` (points to v1.0.0)
- **Concept DOI**: `10.5281/zenodo.1234566` (always points to latest)

For Published REMARKs, use the **version-specific DOI** in CITATION.cff.

### Q: What version number should I use?

**A**: Follow [semantic versioning](https://semver.org/):
- `v1.0.0` - First published release
- `v1.1.0` - New features/functionality added
- `v1.0.1` - Bug fixes, no new features
- `v2.0.0` - Breaking changes

### Q: Can I delete a Zenodo upload?

**A**: No. Zenodo is a permanent archive. However, you can:
- Upload a new version (becomes the latest)
- Hide old versions (but they remain accessible via DOI)

Plan carefully before creating releases.

### Q: How long does Zenodo preserve data?

**A**: Minimum 20 years, but likely indefinitely. Zenodo is operated by CERN and has long-term preservation commitments.

### Q: What if my repository changes after archival?

**A**: The Zenodo archive is frozen at the tagged commit. Your repository can continue to evolve. Create new releases/tags for updated versions.

### Q: Do I need Zenodo for Standard REMARKs?

**A**: No. Zenodo DOI is **only required for Published REMARKs**. Standard REMARKs can be submitted without a DOI.

### Q: Can I use a different archive service?

**A**: Zenodo is preferred for REMARKs because:
- Free and open
- GitHub integration
- Academic-friendly
- CERN-backed (reliable)
- Widely recognized

Other services (Figshare, OSF) can work but require manual steps.

### Q: What's the difference between the econ-ark fork and Zenodo?

**A**: 
- **econ-ark fork**: Live repository that can be updated
- **Zenodo archive**: Frozen snapshot at specific version
- Both should have identical content at the tagged commit

### Q: How do I cite a Zenodo-archived REMARK?

**A**: The CITATION.cff provides the format. Generally:

```
AuthorLastName, F. (YEAR). Title of REMARK.
https://doi.org/10.5281/zenodo.XXXXXXX
```

---

## Example: Complete Workflow

Here's a complete example walkthrough:

```bash
# Step 1: Verify repository is ready
cd ~/my-remark
git status  # Should be clean

# Step 2: Create tag
git tag -a v1.0.0 -m "First published release"
git push origin v1.0.0

# Step 3: Go to Zenodo (via browser)
# - Log in at https://zenodo.org
# - Enable GitHub integration
# - Toggle on your repository

# Step 4: Create GitHub release (via browser)
# - Go to github.com/yourname/your-remark/releases
# - Click "Draft a new release"
# - Select tag v1.0.0
# - Add title: "Version 1.0.0 - Initial Release"
# - Publish

# Step 5: Wait for Zenodo archival (2-10 minutes)
# Check: https://zenodo.org/deposit

# Step 6: Update CITATION.cff with DOI
echo "doi: 10.5281/zenodo.1234567" >> CITATION.cff
git add CITATION.cff
git commit -m "Add Zenodo DOI"
git push

# Step 7: Verify
curl -L -O "https://zenodo.org/records/1234567/files/archive.zip"
unzip archive.zip
cd yourname-your-remark-abc1234
./reproduce.sh --help  # Should work
```

**Total time**: 30-45 minutes from start to finish.

---

## Additional Resources

### Official Documentation
- **Zenodo Help**: https://help.zenodo.org/
- **GitHub + Zenodo Guide**: https://guides.github.com/activities/citable-code/
- **CITATION.cff Spec**: https://citation-file-format.github.io/

### Tools
- **CFF Initializer**: https://citation-file-format.github.io/cff-initializer-javascript/
- **CFF Validator**: https://github.com/citation-file-format/cffconvert

### REMARK Resources
- **STANDARD.md**: Full REMARK requirements
- **How-To-Make-A-REMARK.md**: General REMARK creation guide
- **REMARK Starter Example**: https://github.com/econ-ark/REMARK-starter-example

---

## Summary

### Quick Reference Command Sequence

```bash
# 1. Create tag
git tag -a v1.0.0 -m "Published release"
git push origin v1.0.0

# 2. Enable on Zenodo (browser): https://zenodo.org/account/settings/github/

# 3. Create GitHub release (browser): github.com/you/repo/releases

# 4. Wait for Zenodo archival

# 5. Add DOI to CITATION.cff
# Edit CITATION.cff, add line: doi: 10.5281/zenodo.XXXXXXX

# 6. Commit and push
git add CITATION.cff
git commit -m "Add Zenodo DOI"
git push

# Done!
```

### Timeline

| Phase | Duration | Activity |
|-------|----------|----------|
| Pre-flight | 5 min | Verify repository ready |
| Tagging | 2 min | Create and push git tag |
| Zenodo setup | 5-10 min | Enable integration (first time only) |
| Release | 2 min | Create GitHub release |
| Archival | 5-10 min | Wait for Zenodo processing |
| Update | 2 min | Add DOI to CITATION.cff |
| Verification | 5 min | Download and test archive |
| **Total** | **30-45 min** | **Complete workflow** |

---

## Getting Help

If you encounter issues not covered in this guide:

1. **Check Zenodo documentation**: https://help.zenodo.org/
2. **Search existing issues**: https://github.com/econ-ark/REMARK/issues
3. **Open a new issue**: https://github.com/econ-ark/REMARK/issues/new
4. **Contact REMARK maintainers**: Via GitHub issues

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**Maintained by**: econ-ark REMARK team
