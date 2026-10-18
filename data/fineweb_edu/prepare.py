"""Prepare the byte-level competition corpus from pinned FineWeb-Edu shards.

    python data/fineweb_edu/prepare.py

Downloads three parquet shards of FineWeb-Edu's 10BT sample (about 4.9 GB)
at a fixed dataset revision and checks their SHA-256. Training text comes from
two shards and validation text from a third, so the splits share no documents.
Each document ends with a NUL byte, the byte track's end-of-document marker.
FineWeb-Edu is released under ODC-By 1.0.
"""

import argparse
import hashlib
from pathlib import Path
import urllib.request

from scalerun.training import prepare_data


REVISION = '87f09149ef4734204d70ed1d046ddc9ca3f2b8f9'
URL = f'https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/{REVISION}/sample/10BT/{{name}}'
TRAIN_SHARDS = {
    '000_00000.parquet': 'b1ba7b2ce4cb5ea6ef42dca40263eabb85f37700d01693a68e9b30a31d78e871',
    '001_00000.parquet': '3fcf2dc69cd52503986276d3d2d26a8c356d0f2ea28a0de4fdbda8cf87755693',
}
VAL_SHARDS = {
    '013_00000.parquet': 'b393f51fefab26cd6f4c8f65707c1924f6666c4961a0ebebe04bb57f7ec832de',
}


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def download(name, sha256, cache):
    path = cache / name
    if path.exists() and _sha256(path) == sha256:
        return path
    partial = path.with_suffix('.partial')
    print(f'Downloading {name}')
    with urllib.request.urlopen(URL.format(name=name), timeout=60) as response, partial.open('wb') as out:
        for block in iter(lambda: response.read(1 << 24), b''):
            out.write(block)
    if _sha256(partial) != sha256:
        partial.unlink()
        raise ValueError(f'{name} does not match its pinned SHA-256')
    partial.replace(path)
    return path


def write_text(shards, destination, limit):
    """Write whole documents until the next one would pass limit bytes."""
    import pyarrow.parquet as pq

    written = 0
    with destination.open('wb') as out:
        for shard in shards:
            for batch in pq.ParquetFile(shard).iter_batches(columns=['text'], batch_size=4096):
                for text in batch.column(0).to_pylist():
                    document = text.replace('\0', '').encode('utf-8') + b'\0'
                    if written + len(document) > limit:
                        return written
                    out.write(document)
                    written += len(document)
    raise ValueError(f'shards hold fewer than {limit:,} bytes; lower the limit')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--cache', type=Path, help='Download directory (default: OUTPUT/raw)')
    parser.add_argument('--train_bytes', type=int, default=6_000_000_000)
    parser.add_argument('--val_bytes', type=int, default=10_000_000)
    args = parser.parse_args()
    cache = args.cache or args.output / 'raw'
    cache.mkdir(parents=True, exist_ok=True)
    train = [download(name, sha, cache) for name, sha in TRAIN_SHARDS.items()]
    val = [download(name, sha, cache) for name, sha in VAL_SHARDS.items()]
    print('Extracting text')
    write_text(train, cache / 'train.txt', args.train_bytes)
    write_text(val, cache / 'val.txt', args.val_bytes)
    manifest = prepare_data(cache / 'train.txt', cache / 'val.txt', args.output)
    print(f"Prepared {manifest['train']['tokens']:,} train / {manifest['val']['tokens']:,} validation bytes; "
          f"dataset_id={manifest['dataset_id']}")


if __name__ == '__main__':
    main()
