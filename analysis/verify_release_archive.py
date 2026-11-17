"""Verify ZIP member identities and bytes against the immutable release manifest."""
import hashlib
from pathlib import Path
import zipfile


def verify_archive(archive: Path, release: Path, manifest: dict) -> dict:
    expected = dict(manifest['files_sha256'])
    expected['release_manifest.json'] = hashlib.sha256((release / 'release_manifest.json').read_bytes()).hexdigest()
    prefix = release.name + '/'
    with zipfile.ZipFile(archive) as z:
        members = [i for i in z.infolist() if not i.is_dir()]
        names = [i.filename for i in members]
        if len(names) != len(set(names)) or set(names) != {prefix + n for n in expected}:
            raise RuntimeError('Release ZIP has missing, extra or duplicate file members')
        for info in members:
            name = info.filename[len(prefix):]
            h = hashlib.sha256()
            with z.open(info) as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(chunk)
            if h.hexdigest() != expected[name]:
                raise RuntimeError(f'Release ZIP member hash differs: {name}')
    return {'all_member_bytes_verified': True, 'file_members': len(expected),
            'duplicate_or_extra_members': False, 'manifest_member_verified': True}
