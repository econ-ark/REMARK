from remark.lint import lint_repo, TIER_REQUIREMENTS


def make_repo(path, files, readme_lines=120):
    path.mkdir(parents=True, exist_ok=True)
    for f in files:
        p = path / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('x\n')
    if 'README.md' in files:
        (path / 'README.md').write_text(''.join(f'line {i}\n' for i in range(readme_lines)))
    return path


def test_every_tier_passes_with_its_files(tmp_path):
    for tier, req in TIER_REQUIREMENTS.items():
        repo = make_repo(tmp_path / f'tier{tier}', [str(p) for p in req['required']])
        if tier == 3:
            (repo / 'REMARK.md').write_text('tier: 3\n')
            (repo / 'CITATION.cff').write_text('doi: 10.5281/zenodo.1\n')
        errors, warnings = lint_repo(repo, tier=tier)
        assert errors == [] and warnings == [], (tier, errors, warnings)


def test_missing_files_are_errors(tmp_path):
    repo = make_repo(tmp_path / 'r', ['README.md', 'LICENSE', 'reproduce.sh', 'binder/environment.yml'])
    errors, _ = lint_repo(repo, tier=2)
    assert 'missing Dockerfile' in errors
    assert 'missing REMARK.md' in errors
    assert 'missing CITATION.cff' in errors


def test_short_readme_is_an_error_at_tier_2_but_not_tier_1(tmp_path):
    files = [str(p) for p in TIER_REQUIREMENTS[2]['required']]
    repo = make_repo(tmp_path / 'r', files, readme_lines=91)
    assert 'README.md: 91 lines < 100 required' in lint_repo(repo, tier=2)[0]
    assert lint_repo(repo, tier=1)[0] == []


def test_tier_3_warnings(tmp_path):
    files = [str(p) for p in TIER_REQUIREMENTS[3]['required']]
    repo = make_repo(tmp_path / 'r', files)
    _, warnings = lint_repo(repo, tier=3)
    assert 'REMARK.md: Should specify "tier: 3"' in warnings
    assert any(w.startswith('CITATION.cff: No DOI found') for w in warnings)


def test_optional_files_only_when_asked(tmp_path):
    files = [str(p) for p in TIER_REQUIREMENTS[1]['required']]
    repo = make_repo(tmp_path / 'r', files)
    assert lint_repo(repo, tier=1)[1] == []
    assert lint_repo(repo, tier=1, include_optional=True)[1] == ['optional: reproduce_min.sh not found']
