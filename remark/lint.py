"""Checking a repository against the files STANDARD.md requires at each tier."""

from pathlib import Path

TIER_REQUIREMENTS = {
    1: {  # Docker REMARK
        'required': [
            Path('Dockerfile'),
            Path('reproduce.sh'),
            Path('README.md'),
            Path('LICENSE'),
            Path('binder/environment.yml'),
        ],
        'checks': [
            ('README.md', 50, 'README must be ≥50 lines'),
            ('Dockerfile', None, 'Dockerfile must exist'),
        ]
    },
    2: {  # Reproducible REMARK
        'required': [
            Path('Dockerfile'),
            Path('reproduce.sh'),
            Path('README.md'),
            Path('REMARK.md'),
            Path('CITATION.cff'),
            Path('LICENSE'),
            Path('binder/environment.yml'),
        ],
        'checks': [
            ('README.md', 100, 'README must be ≥100 lines'),
            ('Dockerfile', None, 'Dockerfile must exist'),
            ('REMARK.md', None, 'REMARK.md metadata required'),
            ('CITATION.cff', None, 'CITATION.cff required'),
        ]
    },
    3: {  # Published REMARK (LCD)
        'required': [
            Path('Dockerfile'),
            Path('reproduce.sh'),
            Path('README.md'),
            Path('REMARK.md'),
            Path('CITATION.cff'),
            Path('LICENSE'),
            Path('binder/environment.yml'),
        ],
        'checks': [
            ('README.md', 100, 'README must be comprehensive (≥100 lines)'),
            ('Dockerfile', None, 'Dockerfile must exist'),
            ('REMARK.md', None, 'REMARK.md with tier: 3 metadata'),
            ('CITATION.cff', None, 'CITATION.cff with citation info'),
        ]
    }
}

# Optional files for all tiers
OPTIONAL_FILES = [Path('reproduce_min.sh')]


def lint_repo(local, tier=2, include_optional=False):
    """Check the repository at `local` against a tier. Returns (errors, warnings)."""
    local = Path(local)
    requirements = TIER_REQUIREMENTS[tier]['required']
    checks = TIER_REQUIREMENTS[tier]['checks']
    messages = []
    warnings = []

    # Check required files
    for req in requirements:
        if not local.joinpath(req).exists():
            messages.append(f'missing {req}')

    # Perform additional checks
    for filename, min_lines, message in checks:
        filepath = local / filename
        if filepath.exists():
            if min_lines and filename.endswith('.md'):
                try:
                    with open(filepath, 'r') as f:
                        lines = len([l for l in f.readlines() if l.strip()])
                    if lines < min_lines:
                        messages.append(f'{filename}: {lines} lines < {min_lines} required')
                except Exception as e:
                    warnings.append(f'{filename}: Could not count lines - {e}')

        # Check for tier specification in Tier 3 (DOI now optional/recommended)
        if tier == 3:
            if filename == 'CITATION.cff' and filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if 'doi:' not in content.lower() and '10.' not in content:
                            warnings.append('CITATION.cff: No DOI found (recommended for Tier 3, required for publication)')
                except Exception:
                    pass

            if filename == 'REMARK.md' and filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                        if 'tier: 3' not in content.lower() and 'tier:3' not in content.lower():
                            warnings.append('REMARK.md: Should specify "tier: 3"')
                except Exception:
                    pass

    # Check for optional files
    if include_optional:
        for opt in OPTIONAL_FILES:
            if not local.joinpath(opt).exists():
                warnings.append(f'optional: {opt} not found')

    return messages, warnings
