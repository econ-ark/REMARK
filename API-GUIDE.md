# Programmatic Access Guide for REMARK Repository

This guide documents how to programmatically interact with REMARK data, tools, and standards for AI systems and automated tools.

## Repository Structure for Automated Access

### Key Data Locations
```
REMARK/
├── .ai-context.md          # AI-friendly repository summary
├── schema.json             # Machine-readable schema definitions
├── TOPICS.md               # Structured topic index
├── cli.py                  # Automated CLI tool for REMARK operations
├── REMARKs/                # REMARK metadata catalog
│   ├── *.yml               # Individual REMARK metadata files
│   └── template.md         # Metadata template
├── myst.yml                # Documentation configuration
└── requirements.txt        # Python dependencies
```

## Programmatic Data Access

### REMARK Metadata Format
Each REMARK is described by a YAML file with this structure:
```yaml
name: ProjectName           # Short identifier
remote: https://github.com/org/repo  # Repository URL
title: Human Readable Title # Descriptive title
version: v1.0.0            # Optional: semantic version
```

### Loading REMARK Catalog
```python
import yaml
import os
from pathlib import Path

def load_remark_catalog(catalog_dir="REMARKs"):
    """Load all REMARK metadata from catalog directory."""
    catalog = []
    for yml_file in Path(catalog_dir).glob("*.yml"):
        with open(yml_file, 'r') as f:
            remark_data = yaml.safe_load(f)
            remark_data['metadata_file'] = str(yml_file)
            catalog.append(remark_data)
    return catalog

# Usage
remarks = load_remark_catalog()
print(f"Found {len(remarks)} REMARKs")
```

### Accessing REMARK Schema
```python
import json

def load_remark_schema():
    """Load the JSON schema for REMARK structure."""
    with open('schema.json', 'r') as f:
        return json.load(f)

schema = load_remark_schema()
required_files = schema['definitions']['remarkStandard']['properties']['requiredFiles']['default']
```

## CLI Tool Integration

### Available Commands
The `cli.py` tool provides automated operations:

```bash
# Clone all REMARKs for analysis
python cli.py pull --all

# Lint REMARKs for compliance
python cli.py lint --all

# Build execution environments
python cli.py build conda --all
python cli.py build docker --all

# Execute reproduction scripts
python cli.py execute conda --all

# View logs and results
python cli.py logs
```

### Programmatic CLI Usage
```python
import subprocess
import json

def run_remark_command(command_args):
    """Execute REMARK CLI commands programmatically."""
    cmd = ['python', 'cli.py'] + command_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        'success': result.returncode == 0,
        'stdout': result.stdout,
        'stderr': result.stderr
    }

# Example: Pull a specific REMARK
result = run_remark_command(['pull', 'BufferStockTheory'])
```

## Data Mining and Analysis Patterns

### Repository Analysis
```python
def analyze_remark_topics():
    """Extract topics and patterns from REMARK titles."""
    catalog = load_remark_catalog()
    topics = {}
    
    for remark in catalog:
        title = remark.get('title', '').lower()
        # Extract key terms
        for topic in ['buffer', 'portfolio', 'consumption', 'wealth', 'macro']:
            if topic in title:
                topics[topic] = topics.get(topic, 0) + 1
    
    return topics
```

### Compliance Checking
```python
import requests
from urllib.parse import urlparse

def check_remark_compliance(remark_metadata):
    """Check if a REMARK meets standard requirements."""
    repo_url = remark_metadata.get('remote')
    if not repo_url:
        return {'compliant': False, 'reason': 'No repository URL'}
    
    # Check repository exists and is accessible
    try:
        # GitHub API check
        parsed = urlparse(repo_url)
        if 'github.com' in parsed.netloc:
            api_url = repo_url.replace('github.com', 'api.github.com/repos')
            response = requests.get(api_url)
            repo_info = response.json()
            
            return {
                'compliant': response.status_code == 200,
                'has_releases': len(repo_info.get('releases_url', [])) > 0,
                'last_updated': repo_info.get('updated_at'),
                'stars': repo_info.get('stargazers_count', 0)
            }
    except Exception as e:
        return {'compliant': False, 'error': str(e)}
```

## Integration with External Systems

### GitHub API Integration
```python
def get_remark_repository_info(remark):
    """Get detailed GitHub repository information."""
    repo_url = remark['remote']
    if 'github.com' not in repo_url:
        return None
    
    # Convert to API URL
    api_url = repo_url.replace('github.com', 'api.github.com/repos')
    
    response = requests.get(api_url)
    if response.status_code == 200:
        repo_data = response.json()
        return {
            'name': remark['name'],
            'stars': repo_data.get('stargazers_count', 0),
            'forks': repo_data.get('forks_count', 0),
            'last_updated': repo_data.get('updated_at'),
            'primary_language': repo_data.get('language'),
            'topics': repo_data.get('topics', [])
        }
    return None
```

### MyST Documentation Access
```python
def parse_myst_config():
    """Parse MyST configuration for documentation structure."""
    with open('myst.yml', 'r') as f:
        config = yaml.safe_load(f)
    
    return {
        'title': config['project']['title'],
        'authors': config['project']['authors'],
        'license': config['project']['license'],
        'toc_structure': config['project']['toc'],
        'catalog_url': config.get('catalog', {}).get('url')
    }
```

## Automated Quality Assessment

### REMARK Standard Validation
```python
def validate_remark_standard(repo_path):
    """Validate that a repository meets REMARK standards."""
    required_files = [
        'reproduce.sh',
        'CITATION.cff',
        'binder/environment.yml'
    ]
    
    optional_files = [
        'reproduce_min.sh'
    ]
    
    validation = {
        'compliant': True,
        'missing_required': [],
        'has_optional': [],
        'issues': []
    }
    
    for file in required_files:
        if not os.path.exists(os.path.join(repo_path, file)):
            validation['missing_required'].append(file)
            validation['compliant'] = False
    
    for file in optional_files:
        if os.path.exists(os.path.join(repo_path, file)):
            validation['has_optional'].append(file)
    
    return validation
```

## Bulk Operations and Analysis

### Catalog Statistics
```python
def generate_catalog_statistics():
    """Generate comprehensive statistics about the REMARK catalog."""
    catalog = load_remark_catalog()
    
    stats = {
        'total_remarks': len(catalog),
        'repositories': {
            'github': sum(1 for r in catalog if 'github.com' in r.get('remote', '')),
            'other': sum(1 for r in catalog if 'github.com' not in r.get('remote', ''))
        },
        'topics': analyze_remark_topics(),
        'naming_patterns': {}
    }
    
    # Analyze naming patterns
    for remark in catalog:
        name = remark.get('name', '')
        if name:
            first_word = name.split(/[A-Z]/)[0] if name else ''
            stats['naming_patterns'][first_word] = stats['naming_patterns'].get(first_word, 0) + 1
    
    return stats
```

## Error Handling and Best Practices

### Robust Data Access
```python
def safe_load_remark_data(filepath):
    """Safely load REMARK data with error handling."""
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            # Validate required fields
            required = ['name', 'remote', 'title']
            for field in required:
                if field not in data:
                    return {'error': f'Missing required field: {field}'}
            return data
    except FileNotFoundError:
        return {'error': f'File not found: {filepath}'}
    except yaml.YAMLError as e:
        return {'error': f'YAML parsing error: {e}'}
```

This programmatic access guide enables AI systems and automated tools to effectively interact with the REMARK repository structure, validate compliance, extract metadata, and perform bulk operations on the research catalog. 