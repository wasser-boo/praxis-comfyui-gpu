"""Seed generic API and UI workflows without replacing operator customizations."""
import argparse
from pathlib import Path
import os


def install(source, workspace):
    source, workspace = Path(source), Path(workspace)
    created = []
    for path in sorted(source.glob('*.json')):
        if path.is_symlink() or not path.is_file():
            raise ValueError('Workflow sources must be regular JSON files')
        folders = [workspace / 'workflows']
        if path.name.endswith('-ui.json'):
            folders.append(workspace / 'user/default/workflows')
        for folder in folders:
            for parent in [folder, *folder.parents]:
                if parent == workspace.parent: break
                if parent.is_symlink(): raise ValueError('Refusing a symlink workflow directory')
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / path.name
            if target.exists() or target.is_symlink(): continue
            try:
                with target.open('xb') as output:
                    os.fchmod(output.fileno(), 0o600)
                    output.write(path.read_bytes())
                    output.flush(); os.fsync(output.fileno())
                created.append(str(target))
            except FileExistsError:
                pass  # A concurrent user save wins.
    return created


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='/opt/praxis/workflows')
    parser.add_argument('--workspace', default='/workspace')
    args = parser.parse_args()
    print('Installed', len(install(args.source, args.workspace)), 'missing generic workflows; existing files preserved')
