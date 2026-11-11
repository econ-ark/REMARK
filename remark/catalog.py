"""The catalog: one YAML file per REMARK under REMARKs/, naming its remote, title and tag."""

from dataclasses import dataclass, asdict
from pathlib import Path

from yaml import safe_load


@dataclass
class Metadata:
    remote: str
    local: Path
    yaml: dict
    image_name: str = None

    def __post_init__(self):
        self.image_name = f'econ-ark/{self.local.name.lower().replace(" ", "_")}'

    @property
    def name(self):
        return self.yaml['name']

    @property
    def tag(self):
        return self.yaml.get('tag')

    def flat(self):
        d = {**self.yaml, **asdict(self)}
        del d['yaml']
        return d


def load_catalog(root, repo_home):
    """Map each catalog file, as a path relative to `root`, to its Metadata.

    Each entry's clone lives at `repo_home/<file stem>`.
    """
    root = Path(root)
    metadata = {}
    for p in sorted(root.joinpath('REMARKs').glob('*.yml')):
        with open(p) as f:
            data = safe_load(f)
        data['name'] = p.stem
        metadata[p.relative_to(root)] = Metadata(
            local=Path(repo_home) / data['name'],
            remote=data['remote'],
            yaml=data,
        )
    return metadata
