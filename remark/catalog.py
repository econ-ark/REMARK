"""The catalog: one YAML file per REMARK under REMARKs/, naming its remote, title and tag,
and saying so when the repository is not public yet."""

from dataclasses import dataclass, asdict
from pathlib import Path

from yaml import safe_load


KNOWN_KEYS = ('name', 'remote', 'title', 'tag', 'private')


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

    @property
    def private(self):
        """True for an entry marked `private: true`: its repository is not public yet.
        That the entry exists is public; the website shows nothing of it."""
        return self.yaml.get('private', False)

    @property
    def unknown_keys(self):
        """The entry's keys that the catalog does not define, such as a misspelled `private`."""
        return sorted(key for key in self.yaml if key not in KNOWN_KEYS)

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
        if not isinstance(data.get('private', False), bool):
            raise ValueError(f'{p}: `private` must be true or false, not {data["private"]!r}')
        metadata[p.relative_to(root)] = Metadata(
            local=Path(repo_home) / data['name'],
            remote=data['remote'],
            yaml=data,
        )
    return metadata
