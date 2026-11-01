"""Guard against accidental train/validation/test leakage in temporal pair metadata.

The manifest is the source of truth for which split owns each cloudy patch.
The later-date reference is a proxy target and must stay in that same split.
"""

from pathlib import PurePosixPath


VALID_SPLITS = frozenset({'train', 'val', 'test'})


def check_pair_assignments(manifest_rows, pair_rows):
    """Validate pair-to-manifest split membership before reading raster data.

    Accepts CSV DictReader rows or other mappings. Raises ValueError with the
    offending row number for a missing input, mislabeled split, or bad target.
    """
    manifest_by_path = {row['path']: row for row in manifest_rows}
    if len(manifest_by_path) != len(manifest_rows):
        raise ValueError('manifest.csv contains duplicate patch paths')

    seen_inputs = set()
    for line_number, pair in enumerate(pair_rows, start=2):
        source, target, split = pair['input'], pair['target'], pair['split']
        if split not in VALID_SPLITS:
            raise ValueError(f'pairs.csv line {line_number}: unknown split {split!r}')
        if source in seen_inputs:
            raise ValueError(f'pairs.csv line {line_number}: duplicate input {source!r}')
        seen_inputs.add(source)

        entry = manifest_by_path.get(source)
        if entry is None:
            raise ValueError(f'pairs.csv line {line_number}: input absent from manifest: {source!r}')
        if entry['category'] != 'cloudy':
            raise ValueError(f'pairs.csv line {line_number}: input is not a cloudy patch: {source!r}')
        if entry['split'] != split:
            raise ValueError(
                f'pairs.csv line {line_number}: {source!r} belongs to '
                f"{entry['split']!r}, not {split!r}"
            )

        expected_target = f'data/reference/{split}/{PurePosixPath(source).name}'
        if target != expected_target:
            raise ValueError(
                f'pairs.csv line {line_number}: target {target!r} does not match '
                f'expected same-patch reference {expected_target!r}'
            )
